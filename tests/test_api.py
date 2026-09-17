"""API 后端的离线单测 / Offline tests for API backend.

不依赖网络、不发真 HTTP。覆盖：config 校验、voice cache、SHA-256、sound tag 映射。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_tts.core.config import Config, APIConfig, _str_to_bool
from agentic_tts.engines.api.base import (
    VoiceCache, classify_status,
    APIAuthError, APIServerError, APIClientError, with_retry,
)
from agentic_tts.engines.api.minimax import _normalize_text_for_native_events


# ── Config ──

def test_api_config_defaults():
    api = APIConfig()
    assert api.provider == "minimax"
    assert api.model == "speech-2.8-turbo"
    assert api.fallback_enabled is True
    assert "engine_error" in api.fallback_on_errors


def test_api_config_env_override(monkeypatch):
    monkeypatch.setenv("TTS_API_ENABLED", "true")
    monkeypatch.setenv("TTS_API_PROVIDER", "openai")
    monkeypatch.setenv("TTS_API_KEY", "env-key")
    monkeypatch.setenv("TTS_API_FALLBACK", "false")
    cfg = Config.load()  # 必须 load 才会 _apply_env
    assert cfg.api.enabled is True
    assert cfg.api.provider == "openai"
    assert cfg.api.api_key == "env-key"
    assert cfg.api.fallback_enabled is False


def test_api_provider_validation():
    cfg = Config(api=APIConfig(enabled=True, provider="bogus"))
    with pytest.raises(Exception):
        cfg.validate_runtime()


def test_api_disabled_skips_provider_validation():
    """api.enabled=false 时 provider 错也不报错（避免误启）"""
    cfg = Config(api=APIConfig(enabled=False, provider="bogus"))
    cfg.validate_runtime()  # 不应抛


def test_api_timeout_validation():
    cfg = Config(api=APIConfig(enabled=True, timeout_s=0))
    with pytest.raises(Exception):
        cfg.validate_runtime()


def test_str_to_bool():
    assert _str_to_bool("true") is True
    assert _str_to_bool("yes") is True
    assert _str_to_bool("1") is True
    assert _str_to_bool("false") is False
    assert _str_to_bool("no") is False
    assert _str_to_bool("0") is False
    with pytest.raises(ValueError):
        _str_to_bool("maybe")


# ── VoiceCache ──

def test_voice_cache_hit_and_invalidate(tmp_path):
    cache = VoiceCache(tmp_path / "vc.json")
    key = VoiceCache.make_key("minimax", b"audio-bytes", "transcript", False)
    assert cache.lookup(key) is None
    cache.store(key, voice_id="v_abc123", provider="minimax", ref_text="transcript", x_vector_only=False)
    cache2 = VoiceCache(tmp_path / "vc.json")
    assert cache2.lookup(key) == "v_abc123"


def test_voice_cache_corrupt_recovers(tmp_path):
    p = tmp_path / "vc.json"
    p.write_text("not valid json{", encoding="utf-8")
    cache = VoiceCache(p)
    assert cache._data == {}  # noqa: SLF001
    assert (tmp_path / "vc.json.broken").exists()


def test_voice_cache_key_distinguishes_params():
    a = VoiceCache.make_key("p", b"x", "r1", False)
    b = VoiceCache.make_key("p", b"x", "r2", False)
    c = VoiceCache.make_key("p", b"x", "r1", True)
    assert a != b
    assert a != c


# ── Sound tag mapping ──

def test_sound_tag_known():
    out, warnings = _normalize_text_for_native_events("我笑了 [laugh] 然后 [sigh]")
    assert "(laughs)" in out
    assert "(sighs)" in out
    assert "[laugh]" not in out
    assert any("[laugh]" in w or "(laughs)" in w for w in warnings)


def test_sound_tag_pause_ms():
    out, _ = _normalize_text_for_native_events("前半 [pause:400ms] 后半")
    assert "<#0.40#>" in out


def test_sound_tag_pause_plain():
    out, _ = _normalize_text_for_native_events("A [pause] B")
    assert "<#0.3#>" in out


def test_unknown_sound_kept_as_instruct():
    """未支持的标记（如 [shout]）保留原样 → 引擎层不感知，标记留给 instruct 降级"""
    out, _ = _normalize_text_for_native_events("再喊一遍 [shout]")
    assert "[shout]" in out


# ── Error classification ──

def test_classify_status():
    from agentic_tts.engines.api.base import APIRateLimitError
    assert classify_status(401) is APIAuthError
    assert classify_status(429) is APIRateLimitError
    assert classify_status(503) is APIServerError
    assert classify_status(404).__name__ == "APINotFoundError"


# ── with_retry ──

def test_with_retry_succeeds_after_retries():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise APIServerError("transient")
        return "ok"

    result = with_retry(fn, max_retries=3, backoff_s=0.001, retriable_exceptions=(APIServerError,))
    assert result == "ok"
    assert calls["n"] == 2


def test_with_retry_non_retriable_raises_immediately():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise APIAuthError("no retry")

    with pytest.raises(APIAuthError):
        with_retry(fn, max_retries=3, backoff_s=0.001,
                   retriable_exceptions=(APIServerError,))
    assert calls["n"] == 1


def test_with_retry_gives_up_after_max():
    def fn():
        raise APIServerError("always")

    with pytest.raises(APIServerError):
        with_retry(fn, max_retries=2, backoff_s=0.001, retriable_exceptions=(APIServerError,))