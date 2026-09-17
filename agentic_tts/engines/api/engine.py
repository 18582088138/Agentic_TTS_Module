"""
APITTSEngine —— 把 APIClient 包成门面可用的 TTSEngine。

实现 TTSEngine 抽象：
- declared_capabilities() 静态返回（与 qwen3 引擎对齐）
- synthesize(chunk) 走 client.synthesize；voice_clone 时先 resolve voice_id（命中缓存）
- info() 输出 provider / endpoint / model / 缓存大小 / fallback 状态

按 docs/06_add_engine.md 的四步走 + 6.2 节五条硬约束。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agentic_tts.core.config import Config
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo
from agentic_tts.engines.base import SynthChunk, TTSEngine
from agentic_tts.engines.api.base import (
    APIClient, APISynthRequest, APISynthResult, VoiceCache,
    APIClientError,
)
from agentic_tts.engines.api.minimax import MiniMaxClient

_logger = logging.getLogger("engine.api")

# ─────────────────────────────────────────────────────────
# Audio bytes → numpy float32
# ─────────────────────────────────────────────────────────

def _decode_audio_bytes(audio_bytes: bytes, fmt: str) -> tuple[np.ndarray, int]:
    """
    把 wav/mp3/flac 字节解码成 float32 + sample_rate。
    用 soundfile（项目已有依赖）；失败抛 SynthError。
    """
    import soundfile as sf
    import io as _io
    try:
        data, sr = sf.read(_io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        raise SynthError(f"解码 {fmt} 音频失败：{exc}") from exc
    if data.ndim > 1:
        data = data.mean(axis=1)  # 多声道取平均（接口侧已设 channel=1，这里兜底）
    return np.ascontiguousarray(data.reshape(-1)), int(sr)


# 延迟 import 避免引擎文件触发循环
from agentic_tts.core.errors import SynthError  # noqa: E402


# ─────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────

class APITTSEngine(TTSEngine):
    """通用 API TTS 引擎 / Generic API TTS engine."""

    name = "api"
    codes_per_second: Optional[float] = None        # API 不暴露
    default_max_new_tokens: Optional[int] = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        api_cfg = config.api
        # 选 client
        if api_cfg.provider == "minimax":
            self._client: APIClient = MiniMaxClient(config=config)
        else:
            raise SynthError(
                f"provider={api_cfg.provider!r} 暂未实现　已支持：minimax　"
                "其他 provider 的 adapter 见 docs/10_api_backend.md §17"
            )
        # voice cache（共享 MiniMax 的 cache 实例；或自建）
        self._voice_cache_path = Path(api_cfg.voice_cache_path)
        if not self._voice_cache_path.is_absolute():
            from agentic_tts.core.config import PROJECT_ROOT
            self._voice_cache_path = PROJECT_ROOT / self._voice_cache_path
        # 解析 ref_audio 的本地辅助
        self._fallback_count: int = 0
        _logger.info("APITTSEngine ready · provider=%s model=%s",
                     api_cfg.provider, api_cfg.model)

    # ── 能力声明（与 qwen3 对齐）──

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.CUSTOM_VOICE,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.INSTRUCT,
            # VOCAL_EVENTS 由 MiniMax 2.8 原生支持（映射见 _normalize_text_for_native_events）
            Capability.VOCAL_EVENTS,
            # BATCH 第一版不声明
        }

    # ── 内置音色（用 MiniMax 系统音色表，避免依赖网络）──
    # 静态打底；用户可写 voices.yaml 自定义
    _BUILTIN_SPEAKERS = [
        ("male-qn-qingse", "Chinese", "清澈男声"),
        ("male-qn-jingying", "Chinese", "精英男声"),
        ("female-shaonv", "Chinese", "少女女声"),
        ("female-yujie", "Chinese", "御姐女声"),
        ("English_expressive_narrator", "English", "Expressive narrator"),
        ("English_lady_female", "English", "English lady"),
    ]

    def builtin_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(name=n, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      language=lang, description=desc, speaker=n)
            for n, lang, desc in self._BUILTIN_SPEAKERS
        ]

    def languages(self) -> list[str]:
        return ["Chinese", "English", "Japanese", "Korean",
                "Spanish", "French", "German", "auto"]

    # ── 主入口：synthesize ──

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        self.require_mode(chunk.mode)

        # voice_clone：先用 ref_audio 找/建 voice_id
        voice_id: Optional[str] = None
        warnings: list[str] = []
        if chunk.mode is Mode.VOICE_CLONE:
            voice_id, voice_warnings = self._resolve_voice_id(chunk)
            warnings.extend(voice_warnings)

        # 把 Mode 映射成 provider mode 字符串
        provider_mode = {
            Mode.CUSTOM_VOICE: "custom_voice",
            Mode.VOICE_DESIGN: "voice_design",
            Mode.VOICE_CLONE: "voice_clone",
        }[chunk.mode]

        req = APISynthRequest(
            text=chunk.text,
            mode=provider_mode,
            language=chunk.language,
            speaker=chunk.speaker if chunk.mode is Mode.CUSTOM_VOICE else None,
            instruct=chunk.instruct,
            voice_id=voice_id,
            ref_text=chunk.ref_text if chunk.x_vector_only is False else None,
            sample_rate=24000,
            audio_format="wav",
            gen=dict(chunk.gen or {}),
            seed=chunk.seed,
        )

        started = time.perf_counter()
        try:
            result: APISynthResult = self._client.synthesize(req)
        except APIClientError as exc:
            # 已分类的错误；保留 actionable 信息
            raise SynthError(str(exc)) from exc
        elapsed = time.perf_counter() - started

        # 解码音频字节
        wav, sr = _decode_audio_bytes(result.audio_bytes, result.audio_format)
        seconds = len(wav) / sr if sr else 0.0
        warnings.extend(result.warnings or [])

        return RawAudio(
            wav=wav,
            sample_rate=sr,
            effective_gen={
                "provider": self.config.api.provider,
                "model": self.config.api.model,
                "voice_id_used": result.voice_id_used,
                "generate_seconds": round(elapsed, 2),
                "audio_format": result.audio_format,
            },
            instruct_applied=bool(chunk.instruct),
            warnings=warnings,
        )

    # ── voice_id 解析（带缓存）──

    def _resolve_voice_id(self, chunk: SynthChunk) -> tuple[str, list[str]]:
        if not chunk.ref_audio:
            raise SynthError("voice_clone 需要 ref_audio")
        path = self._resolve_ref_audio(chunk.ref_audio)
        audio_bytes = Path(path).read_bytes()
        warnings: list[str] = []
        # x_vector_only 不上传 ref_text 字段，但 cache key 仍按是否带 ref_text 区分
        voice_id = self._client.upload_clone_audio(
            audio_bytes, ref_text=chunk.ref_text, voice_id_hint=None
        )
        return voice_id, warnings

    @staticmethod
    def _resolve_ref_audio(ref_audio: str) -> str:
        """解析本地路径；拒绝 URL（沿用 local 引擎约定）。"""
        from agentic_tts.core.errors import SynthError
        if ref_audio.startswith(("http://", "https://")):
            raise SynthError("API 引擎的 ref_audio 不支持 URL，请用本地路径或 base64")
        if ref_audio.startswith("data:audio"):
            import base64, tempfile
            header, _, payload = ref_audio.partition(",")
            suffix = ".wav" if "wav" in header else ".mp3"
            handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            handle.write(base64.b64decode(payload))
            handle.close()
            return handle.name
        path = Path(ref_audio)
        if not path.is_file():
            raise SynthError(f"ref_audio 不存在：{ref_audio}")
        return str(path)

    # ── 元信息 ──

    def info(self) -> dict[str, Any]:
        cache_size = 0
        try:
            cache = VoiceCache(self._voice_cache_path)
            cache_size = len(cache._data)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        return {
            **super().info(),
            "provider": self.config.api.provider,
            "endpoint": self._client.base_url if hasattr(self._client, "base_url") else "",
            "model": self.config.api.model,
            "voice_cache_path": str(self._voice_cache_path),
            "voice_cache_size": cache_size,
            "fallback_enabled": self.config.api.fallback_enabled,
            "fallback_count": self._fallback_count,
        }

    def release(self) -> None:
        # API 引擎无显存；纯函数调用
        pass


__all__ = ["APITTSEngine"]