"""
MiniMax TTS API adapter.

Endpoints (按官方文档 2026-Q3):
  - POST {base}/v1/t2a_v2         同步合成
  - POST {base}/v1/voice_clone    创建克隆音色
  - POST {base}/v1/text2voice     文生音色（voice_design）
  - POST {base}/v1/files/upload   上传参考音频（multipart）

鉴权: Authorization: Bearer <api_key>
       GroupId: <group_id> (header，某些场景)

API key 解析顺序：config.api.api_key → 环境变量 MINIMAX_API_KEY → 抛 APIAuthError
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agentic_tts.core.config import Config
from agentic_tts.core.errors import ConfigError
from agentic_tts.engines.api.base import (
    APIClient, APISynthRequest, APISynthResult, VoiceCache,
    APIClientError, APIAuthError, APIRateLimitError, APIServerError,
    APITimeoutError, APINotFoundError, APIBadRequestError,
    make_error, with_retry,
)

_logger = logging.getLogger("api.minimax")

# MiniMax 文档里的 base_url，二选一（看你账号所在区域）
_DEFAULT_BASE = "https://api.minimax.io"  # 按 2026 官方文档

# 2.8 原生 sound tags → 文档列出的 (xxx) 形式
# 我们的 [laugh] [sigh] 等事件标记映射到这些
_SOUND_TAG_MAP = {
    "laugh": "(laughs)",
    "chuckle": "(chuckle)",
    "cough": "(coughs)",
    "sigh": "(sighs)",
    "breath": "(breath)",
    "cry": "(crying)",
    "sniff": "(sniffs)",
    "humming": "(humming)",
    "whistle": "(whistles)",
    # 没有官方对应的（如 shout）：保留 instruct 降级
}

# [pause:Xs] → MiniMax 文档的 <#X#>
_PAUSE_RE = re.compile(r"\[pause:(\d+(?:\.\d+)?)(ms|s)\]|\[pause\]")


def _normalize_text_for_native_events(text: str) -> tuple[str, list[str]]:
    """
    把 [laugh]/[pause:400ms] 翻译成 MiniMax 2.8 原生标记。
    返回 (normalized_text, warnings)。
    """
    warnings: list[str] = []
    # pause 标记
    def _pause_sub(m: re.Match) -> str:
        val = m.group(1)
        unit = m.group(2)
        if val is None:  # [pause] 无时长
            return "<#0.3#>"
        if unit == "ms":
            return f"<#{float(val)/1000:.2f}#>"
        return f"<#{val}#>"
    out = _PAUSE_RE.sub(_pause_sub, text)

    # 情感标记 → 原生
    for tag, native in _SOUND_TAG_MAP.items():
        pattern = re.compile(rf"\[{tag}\]", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(native, out)
            warnings.append(f"[{tag}] → {native} (MiniMax 2.8 原生)")
    return out, warnings


@dataclass
class MiniMaxClient:
    """MiniMax API 客户端 / MiniMax API client."""

    name: str = "minimax"
    config: Config = field(default=None)  # type: ignore[assignment]

    base_url: str = ""
    _api_key: str = ""
    _group_id: str = ""
    _cache: Optional[VoiceCache] = None

    def __post_init__(self) -> None:
        api = self.config.api
        self.base_url = api.endpoint or _DEFAULT_BASE
        self.base_url = self.base_url.rstrip("/")
        # 鉴权三源：config → env TTS_API_KEY → MINIMAX_API_KEY
        key = api.api_key or os.environ.get("TTS_API_KEY") or os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise APIAuthError(
                "MiniMax API key 未配置　设环境变量 MINIMAX_API_KEY 或在 configs/config.yaml 的 api.api_key 填"
            )
        self._api_key = key
        self._group_id = api.group_id or os.environ.get("MINIMAX_GROUP_ID", "")
        # voice cache
        cache_path = Path(api.voice_cache_path)
        if not cache_path.is_absolute():
            from agentic_tts.core.config import PROJECT_ROOT
            cache_path = PROJECT_ROOT / cache_path
        self._cache = VoiceCache(cache_path)
        _logger.info("MiniMax client ready (endpoint=%s, model=%s)", self.base_url, api.model)

    # ── HTTP 层（httpx 在函数体内 import，避免启动期拉重依赖）──

    def _post(self, path: str, json_body: dict, *, timeout: int = 60) -> dict:
        """POST 一段 JSON；走重试与错误分类。"""
        import httpx

        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._group_id:
            headers["GroupId"] = self._group_id

        api = self.config.api
        retriable = (APIServerError, APITimeoutError, APIRateLimitError)

        def _call() -> dict:
            try:
                resp = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
            except httpx.TimeoutException as exc:
                raise APITimeoutError(f"POST {path} 超时: {exc}") from exc
            except httpx.HTTPError as exc:
                # 网络层错 → 当 server error 重试
                raise APIServerError(f"POST {path} 网络错: {exc}") from exc

            if resp.status_code >= 400:
                err = make_error(resp.status_code, resp.text, url)
                # 限流/服务错 → 重试；其余 → 直接抛
                if not isinstance(err, retriable):
                    raise err
                raise err

            try:
                return resp.json()
            except Exception as exc:
                raise APIBadRequestError(f"响应非 JSON: {resp.text[:200]}") from exc

        return with_retry(_call, max_retries=api.max_retries,
                          backoff_s=api.retry_backoff_s, retriable_exceptions=retriable)

    # ── 合成主入口 ──

    def synthesize(self, req: APISynthRequest) -> APISynthResult:
        """走 /v1/t2a_v2。"""
        normalized, native_warnings = _normalize_text_for_native_events(req.text)

        if req.mode == "voice_clone" and not req.voice_id:
            raise ConfigError("voice_clone 需要 voice_id（先经过 upload_clone_audio）")

        body: dict[str, Any] = {
            "model": self.config.api.model,
            "text": normalized,
            "stream": False,
            "language_boost": _to_language_boost(req.language),
            "output_format": "hex",                  # hex 编码音频字节（最通用）
            "voice_setting": {
                "voice_id": req.voice_id or req.speaker or "male-qn-qingse",
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": req.sample_rate,
                "bitrate": 128000,
                "format": req.audio_format,
                "channel": 1,
            },
        }
        if req.instruct:
            body["voice_setting"]["instruct"] = req.instruct

        resp = self._post("/v1/t2a_v2", body, timeout=self.config.api.timeout_s)
        _check_base_resp(resp)

        # hex 字节 → bytes
        audio_hex = resp.get("audio_file", "")
        try:
            audio_bytes = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise APIBadRequestError(f"audio_file 非 hex: {audio_hex[:60]}…") from exc

        extra = resp.get("extra_info") or {}
        seconds = float(extra.get("audio_length", 0.0))
        sample_rate = int(extra.get("audio_sample_rate", req.sample_rate))

        return APISynthResult(
            audio_bytes=audio_bytes,
            audio_format=req.audio_format,
            sample_rate=sample_rate,
            seconds=seconds,
            voice_id_used=req.voice_id or req.speaker,
            warnings=native_warnings,
        )

    # ── voice_clone 流程 ──

    def upload_clone_audio(self, audio_bytes: bytes, *, ref_text: Optional[str] = None,
                            voice_id_hint: Optional[str] = None) -> str:
        """
        1. POST /v1/files/upload（multipart）→ file_id
        2. POST /v1/voice_clone（带 file_id + voice_id）→ 拿到 voice_id
        3. 写入本地 cache

        voice_id_hint 用来让生成的 voice_id 可读（如 'clone_default_08'），不传就生成 hash 前缀。
        """
        api = self.config.api
        cache_key = VoiceCache.make_key(self.name, audio_bytes, ref_text, x_vector_only=False)
        assert self._cache is not None
        cached = self._cache.lookup(cache_key)
        if cached:
            _logger.info("voice cache HIT key=%s voice_id=%s", cache_key[:32], cached)
            return cached

        file_id = self._upload_file(audio_bytes, purpose="voice_clone")
        voice_id = voice_id_hint or f"clone_{hashlib_short(audio_bytes)}_{int(time.time())%10**6:06d}"
        # 名称规则：英文字母开头、字母/数字/-/_
        voice_id = re.sub(r"[^A-Za-z0-9_-]", "_", voice_id)[:64]
        if not re.match(r"^[A-Za-z]", voice_id):
            voice_id = "v_" + voice_id

        body = {
            "file_id": file_id,
            "voice_id": voice_id,
            "model": api.model,
            "text": ref_text or "这是用于试听的声音克隆样本。",
            "language_boost": "auto",
            "need_noise_reduction": False,
            "need_volume_normalization": False,
        }
        resp = self._post("/v1/voice_clone", body, timeout=api.timeout_s)
        _check_base_resp(resp)
        # resp 通常含 voice_id 回显与 trial_audio；以请求里的 voice_id 为准
        self._cache.store(cache_key, voice_id=voice_id, provider=self.name,
                          ref_text=ref_text, x_vector_only=False)
        _logger.info("voice clone OK key=%s voice_id=%s", cache_key[:32], voice_id)
        return voice_id

    def _upload_file(self, audio_bytes: bytes, *, purpose: str = "voice_clone") -> int:
        """POST /v1/files/upload；返回 file_id。"""
        import httpx

        url = f"{self.base_url}/v1/files/upload"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._group_id:
            headers["GroupId"] = self._group_id
        files = {"file": ("ref_audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")}
        data = {"purpose": purpose}

        try:
            resp = httpx.post(url, headers=headers, files=files, data=data,
                              timeout=self.config.api.timeout_s)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"file upload 超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise APIServerError(f"file upload 网络错: {exc}") from exc

        if resp.status_code >= 400:
            raise make_error(resp.status_code, resp.text, url)

        try:
            payload = resp.json()
        except Exception as exc:
            raise APIBadRequestError(f"file upload 响应非 JSON: {resp.text[:200]}") from exc

        file_obj = payload.get("file") or {}
        file_id = file_obj.get("file_id")
        if not file_id:
            raise APIBadRequestError(f"file upload 没返回 file_id: {payload}")
        _logger.info("file uploaded id=%s", file_id)
        return int(file_id)


def hashlib_short(data: bytes, n: int = 8) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()[:n]


def _to_language_boost(language: str) -> str:
    """把我们的语言名映射到 MiniMax 的 language_boost 值。"""
    m = {
        "Chinese": "Chinese", "English": "English", "Japanese": "Japanese",
        "Korean": "Korean", "auto": "auto",
    }
    return m.get(language, "auto")


def _check_base_resp(resp: dict) -> None:
    """MiniMax 响应里 base_resp.status_code != 0 表示业务错误。"""
    base = resp.get("base_resp") or {}
    code = base.get("status_code", 0)
    if code != 0:
        msg = base.get("status_msg", "")
        raise APIBadRequestError(
            f"MiniMax 业务错 status_code={code} msg={msg}\n　"
            f"（常见 1002=RPM 限流 1004=鉴权失败 1008=余额不足 1013=内部 1039=TPM 限流）"
        )


def register() -> None:
    """注册到全局 adapter 表（如果存在）。"""
    # 留 hook —— 真实注册在 __init__ 里按需触发，避免 import-time 副作用
    pass


__all__ = ["MiniMaxClient", "_normalize_text_for_native_events", "register"]