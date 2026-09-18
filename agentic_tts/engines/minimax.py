"""
MiniMax TTS 引擎（HTTP API）/ MiniMax TTS engine via HTTP API.

与 qwen3.py 完全同构：实现 synthesize() + declared_capabilities()，不做文本预处理/分段/拼接（门面负责）。
不声明 VOCAL_EVENTS（MiniMax 2.8 有原生 sound tag 但本实现第一版不解析，留作后续扩展）。
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from agentic_tts.core.config import Config
from agentic_tts.core.errors import EngineError, SynthError
from agentic_tts.core.registry import register_engine
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo
from agentic_tts.engines.base import SynthChunk, TTSEngine

_logger = logging.getLogger("engine.minimax")

# 按 key 前缀选 endpoint
_ENDPOINT_BY_KEY = {
    "sk-cp-": "https://api.minimaxi.com",   # 国内 Token Plan
}
_DEFAULT_ENDPOINT = "https://api.minimax.io"  # 海外按量付费


def _pick_endpoint(api_key: str, configured: str) -> str:
    if configured:
        return configured.rstrip("/")
    for prefix, ep in _ENDPOINT_BY_KEY.items():
        if api_key.startswith(prefix):
            return ep
    return _DEFAULT_ENDPOINT


def _resolve_ref_audio(ref_audio: str) -> str:
    """拒绝 URL；base64 落临时文件；本地路径校验存在。"""
    if ref_audio.startswith(("http://", "https://")):
        raise SynthError("API 引擎 ref_audio 不支持 URL，请用本地路径")
    if ref_audio.startswith("data:audio"):
        import tempfile
        header, _, payload = ref_audio.partition(",")
        suffix = ".wav" if "wav" in header else ".mp3"
        h = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        h.write(base64.b64decode(payload))
        h.close()
        return h.name
    p = Path(ref_audio)
    if not p.is_file():
        raise SynthError(f"ref_audio 不存在：{ref_audio}")
    return str(p)


def _audio_bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@register_engine("minimax")
class MiniMaxEngine(TTSEngine):
    """MiniMax TTS API 引擎。"""

    name = "minimax"
    codes_per_second: Optional[float] = None   # 不暴露
    default_max_new_tokens: Optional[int] = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        api = config.api
        key = api.api_key or _get_api_key_from_env()
        if not key:
            raise EngineError(
                "MiniMax API key 未配置：设环境变量 TTS_API_KEY 或在 configs/config.yaml 的 api.api_key 填"
            )
        self._api_key = key
        self.base_url = _pick_endpoint(key, api.endpoint)
        self.model = api.model
        self.timeout_s = api.timeout_s
        self._cache_path = Path("voices/api_voice_cache.json")  # gitignored
        self._cache = self._load_cache()
        _logger.info("MiniMax engine ready: endpoint=%s model=%s", self.base_url, self.model)

    # ── 能力 / capabilities ──

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.CUSTOM_VOICE,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.INSTRUCT,
        }

    def builtin_voices(self) -> list[VoiceInfo]:
        # 静态打底；用户可在 voices.yaml 自定义
        system_voices = [
            ("male-qn-qingse", "Chinese"),
            ("male-qn-jingying", "Chinese"),
            ("female-shaonv", "Chinese"),
            ("English_expressive_narrator", "English"),
        ]
        return [
            VoiceInfo(name=n, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      language=lang, description="", speaker=n)
            for n, lang in system_voices
        ]

    def languages(self) -> list[str]:
        return ["Chinese", "English", "Japanese", "Korean", "auto"]

    # ── 合成 / synthesis ──

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        self.require_mode(chunk.mode)

        voice_id = chunk.speaker or "male-qn-qingse"
        if chunk.mode is Mode.VOICE_CLONE:
            voice_id = self._resolve_or_clone_voice(chunk)

        body = self._build_request(chunk, voice_id)

        started = time.perf_counter()
        try:
            import httpx
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(
                    f"{self.base_url}/v1/t2a_v2",
                    headers={"Authorization": f"Bearer {self._api_key}",
                             "Content-Type": "application/json"},
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise SynthError(f"MiniMax 调用失败：{exc}") from exc
        elapsed = time.perf_counter() - started

        if (data.get("base_resp") or {}).get("status_code", 0) != 0:
            msg = data["base_resp"].get("status_msg", "")
            raise SynthError(f"MiniMax 业务错：{msg}")

        # 真实字段是 data.audio（hex 编码）
        hex_audio = (data.get("data") or {}).get("audio", "")
        try:
            audio_bytes = bytes.fromhex(hex_audio)
        except ValueError as exc:
            raise SynthError(f"data.audio 非 hex：{hex_audio[:60]}…") from exc

        wav = self._decode_audio(audio_bytes, body["audio_setting"]["format"])
        sample_rate = int((data.get("extra_info") or {}).get("audio_sample_rate",
                                                             body["audio_setting"]["sample_rate"]))
        seconds = len(wav) / sample_rate if sample_rate else 0.0

        return RawAudio(
            wav=wav,
            sample_rate=sample_rate,
            effective_gen={
                "provider": "minimax",
                "model": self.model,
                "voice_id_used": voice_id,
                "generate_seconds": round(elapsed, 2),
            },
            instruct_applied=bool(chunk.instruct),
        )

    # ── voice clone + cache ──

    def _resolve_or_clone_voice(self, chunk: SynthChunk) -> str:
        """先查 SHA-256 缓存，命中返回；未命中上传+克隆，写缓存。"""
        if not chunk.ref_audio:
            raise SynthError("voice_clone 需要 ref_audio")
        path = _resolve_ref_audio(chunk.ref_audio)
        audio_bytes = Path(path).read_bytes()
        digest = _audio_bytes_hash(audio_bytes)
        cache_key = f"{digest}|{chunk.ref_text or ''}|{int(chunk.x_vector_only)}"

        cached = self._cache.get(cache_key)
        if cached:
            _logger.info("voice cache HIT voice_id=%s", cached["voice_id"])
            return cached["voice_id"]

        voice_id = self._do_clone(audio_bytes, chunk, digest)
        self._cache[cache_key] = {
            "voice_id": voice_id,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_cache()
        _logger.info("voice clone OK voice_id=%s", voice_id)
        return voice_id

    def _do_clone(self, audio_bytes: bytes, chunk: SynthChunk, digest: str) -> str:
        """1. upload file → file_id  2. POST voice_clone → voice_id"""
        import httpx

        # 生成 voice_id（要求英文字母开头，仅含字母数字-_）
        vid = f"v{digest[:8]}_{int(time.time()) % 10**6:06d}"[:64]

        with httpx.Client(timeout=self.timeout_s) as client:
            # 1) 上传文件
            files = {"file": ("ref.mp3", audio_bytes, "audio/mpeg")}
            data = {"purpose": "voice_clone"}
            r = client.post(
                f"{self.base_url}/v1/files/upload",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files=files, data=data,
            )
            r.raise_for_status()
            file_id = r.json()["file"]["file_id"]

            # 2) 克隆
            body: dict = {
                "file_id": file_id, "voice_id": vid, "model": self.model,
                "language_boost": "auto",
                "need_noise_reduction": False,
                "need_volume_normalization": False,
            }
            if chunk.ref_text:
                body["text"] = chunk.ref_text
            r = client.post(
                f"{self.base_url}/v1/voice_clone",
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            return vid

    def _build_request(self, chunk: SynthChunk, voice_id: str) -> dict:
        """按 MiniMax 官方格式构造 t2a_v2 请求。"""
        voice_setting = {"voice_id": voice_id, "speed": 1.0, "vol": 1.0, "pitch": 0}
        if chunk.instruct:
            voice_setting["instruct"] = chunk.instruct
        return {
            "model": self.model,
            "text": chunk.text,
            "stream": False,
            "language_boost": _to_language_boost(chunk.language),
            "output_format": "hex",
            "voice_setting": voice_setting,
            "audio_setting": {"sample_rate": 24000, "bitrate": 128000,
                              "format": "wav", "channel": 1},
        }

    def _decode_audio(self, audio_bytes: bytes, fmt: str) -> np.ndarray:
        import io as _io
        import soundfile as sf
        try:
            data, _ = sf.read(_io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            raise SynthError(f"解码 {fmt} 音频失败：{exc}") from exc
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data.reshape(-1))

    # ── voice cache 持久化 ──

    def _load_cache(self) -> dict:
        if not self._cache_path.is_file():
            return {}
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _to_language_boost(language: str) -> str:
    return {"Chinese": "Chinese", "English": "English",
            "Japanese": "Japanese", "Korean": "Korean"}.get(language, "auto")


def _get_api_key_from_env() -> Optional[str]:
    import os
    return os.environ.get("TTS_API_KEY") or os.environ.get("MINIMAX_API_KEY")


# 引擎注册
__all__ = ["MiniMaxEngine"]