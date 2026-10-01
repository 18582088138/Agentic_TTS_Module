"""
MiniMax 账号音色并入内置列表 —— 假 httpx，零网络、零费用。

复测 / Re-run:
    python -m pytest tests/test_minimax_voices.py -q

覆盖什么（docs/issues/006）:
  · 账号里的克隆音色出现在 builtin_voices，且 VoiceStore 按名字能解析成 custom_voice
  · 查不到账号音色时退回静态系统音色，不抛错（服务照常起）
"""

from __future__ import annotations

import httpx
import pytest

from agentic_tts.core.config import Config
from agentic_tts.core.errors import ConfigError
from agentic_tts.core.types import Mode, SynthRequest
from agentic_tts.engines.minimax import MiniMaxEngine
from agentic_tts.voices.store import VoiceStore

CLONED = "va0d7850d_790636"


class _Resp:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TTS_API_KEY", "sk-cp-fake")
    return MiniMaxEngine(Config())


def test_account_cloned_voice_is_selectable(engine, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp({
        "voice_cloning": [{"voice_id": CLONED, "created_time": "2026-09-19"}],
        "base_resp": {"status_code": 0},
    }))
    store = VoiceStore(Config(), builtin=engine.builtin_voices())

    resolved = store.resolve(SynthRequest(text="x", voice=CLONED))

    assert resolved.mode is Mode.CUSTOM_VOICE
    assert resolved.speaker == CLONED


def test_listing_failure_falls_back_to_static_voices(engine, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", boom)
    names = [v.name for v in engine.builtin_voices()]

    assert "male-qn-qingse" in names
    with pytest.raises(ConfigError):
        VoiceStore(Config(), builtin=engine.builtin_voices()).resolve(
            SynthRequest(text="x", voice=CLONED))
