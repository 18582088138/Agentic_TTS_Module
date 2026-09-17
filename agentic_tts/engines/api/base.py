"""
API client 抽象层 / API client abstraction.

`APIClient` 是与 provider 解耦的薄协议 —— 换供应商（MiniMax ↔ DashScope ↔ OpenAI）
只换 adapter 文件，调用方不动。

共享能力：
- 鉴权 / endpoint 解析
- 重试（指数退避）+ 错误分类
- voice_id 缓存（SHA-256 内容 hash + provider + ref_text + x_vector_only）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from agentic_tts.core.errors import EngineError, SynthError

_logger = logging.getLogger("api.client")


# ─────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────

@dataclass
class APISynthRequest:
    """门面交给 adapter 的一次合成请求 / One synthesis request to the adapter."""

    text: str
    mode: str                       # custom_voice | voice_design | voice_clone
    language: str = "Chinese"
    speaker: Optional[str] = None   # 系统 voice_id 或内置音色名
    instruct: Optional[str] = None
    voice_id: Optional[str] = None  # 已上传/缓存的克隆 voice_id（voice_clone 时用）
    ref_text: Optional[str] = None  # ICL 模式时给上游
    sample_rate: int = 24000
    audio_format: str = "wav"       # wav | mp3 | flac
    gen: dict[str, Any] = None      # 引擎私有参数透传
    seed: int = 0


@dataclass
class APISynthResult:
    """adapter 返回的合成结果 / Adapter's synth result."""

    audio_bytes: bytes              # 原始音频字节（wav/mp3/flac）
    audio_format: str               # 实际格式
    sample_rate: int
    seconds: float
    voice_id_used: Optional[str] = None
    warnings: list[str] = None      # type: ignore[assignment]


class APIClient(Protocol):
    """Provider 解耦的协议 / Provider-agnostic protocol."""

    name: str

    def synthesize(self, req: APISynthRequest) -> APISynthResult: ...

    # voice_clone 流程：upload → clone → 拿 voice_id；已上传的由 cache 层处理
    def upload_clone_audio(self, audio_bytes: bytes, *, ref_text: Optional[str] = None,
                           voice_id_hint: Optional[str] = None) -> str: ...


# ─────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────

# HTTP 状态码 → 错误分类的映射（适配器用）
class APIClientError(EngineError):
    """API 客户端通用错误。"""


class APIAuthError(APIClientError):
    """401 / 403 — 鉴权失败。"""


class APIRateLimitError(APIClientError):
    """429 — 限流。"""


class APIServerError(APIClientError):
    """5xx — 上游故障。"""


class APITimeoutError(APIClientError):
    """请求超时。"""


class APINotFoundError(APIClientError):
    """404 — endpoint 错。"""


class APIBadRequestError(APIClientError):
    """400 — 业务错误，原始信息透传。"""


def classify_status(status: int, body: Optional[str] = None) -> type[APIClientError]:
    """根据 HTTP 状态码选错误类。"""
    if status in (401, 403):
        return APIAuthError
    if status == 404:
        return APINotFoundError
    if status == 429:
        return APIRateLimitError
    if status >= 500:
        return APIServerError
    return APIBadRequestError


def make_error(status: int, body: Optional[str], endpoint: str) -> APIClientError:
    """生成带操作建议的错误实例 / Build a typed error with actionable hint."""
    cls = classify_status(status, body)
    body_excerpt = (body or "")[:300].strip()
    hints = {
        APIAuthError: "鉴权失败：检查 TTS_API_KEY 是否对、是否过期",
        APIRateLimitError: "被限流：降低调用频率或联系 provider 提配额",
        APIServerError: "上游故障：稍后重试，或检查 provider 状态页",
        APITimeoutError: "请求超时：调小 text 长度，或调大 api.timeout_s",
        APINotFoundError: f"端点不存在：检查 api.endpoint / api.provider（当前 endpoint={endpoint}）",
        APIBadRequestError: "请求被上游拒绝：检查参数或业务错误码",
    }
    msg = f"[{status}] {body_excerpt}\n　{hints.get(cls, '')}"
    return cls(msg)


# ─────────────────────────────────────────────────────────
# Voice cache (SHA-256 内容 hash)
# ─────────────────────────────────────────────────────────

class VoiceCache:
    """
    voice_id 本地缓存 / Local voice_id cache.

    key = sha256(ref_audio_bytes) + "|" + provider + "|" + ref_text + "|" + x_vector_only
    value = { voice_id, provider, ref_text, x_vector_only, uploaded_at }

    文件不存在视为空；损坏自动备份成 .broken 并重建。
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("cache 顶层不是 dict")
            self._data = raw
        except Exception as exc:  # noqa: BLE001
            broken = self.path.with_suffix(self.path.suffix + ".broken")
            try:
                self.path.rename(broken)
            except OSError:
                pass
            _logger.warning("voice cache %s 损坏（%s），已备份成 %s，从空开始", self.path, exc, broken)
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def make_key(provider: str, audio_bytes: bytes, ref_text: Optional[str],
                 x_vector_only: bool) -> str:
        digest = hashlib.sha256(audio_bytes).hexdigest()
        return f"{provider}|{digest}|{ref_text or ''}|{int(x_vector_only)}"

    def lookup(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        return entry.get("voice_id") if entry else None

    def store(self, key: str, *, voice_id: str, provider: str, ref_text: Optional[str],
              x_vector_only: bool) -> None:
        self._data[key] = {
            "voice_id": voice_id,
            "provider": provider,
            "ref_text": ref_text,
            "x_vector_only": x_vector_only,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)
        self.save()


# ─────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────

def with_retry(fn, *, max_retries: int, backoff_s: float,
               retriable_exceptions: tuple[type[BaseException], ...]) -> Any:
    """
    指数退避重试 / Exponential-backoff retry.

    只对 retriable_exceptions 重试；其他异常（如 APIAuthError）直接抛。
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retriable_exceptions as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = backoff_s * (2 ** attempt)
            _logger.warning("retry %d/%d after %s: %s", attempt + 1, max_retries,
                            type(exc).__name__, exc)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────
# Secret env helper（本地开发用 .env / CI 用真 env）
# ─────────────────────────────────────────────────────────

def load_secret_from_env(name: str, *, dotenv_path: Optional[Path] = None) -> Optional[str]:
    """
    先看环境变量；空就尝试读 .env 文件。
    Returns None 表示没找到（**绝不抛错**，由调用方决定是否报错）。
    """
    val = os.environ.get(name)
    if val:
        return val
    if dotenv_path and dotenv_path.is_file():
        try:
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return None


__all__ = [
    "APIClient", "APISynthRequest", "APISynthResult",
    "VoiceCache", "with_retry", "load_secret_from_env",
    "APIClientError", "APIAuthError", "APIRateLimitError",
    "APIServerError", "APITimeoutError", "APINotFoundError",
    "APIBadRequestError", "classify_status", "make_error",
]