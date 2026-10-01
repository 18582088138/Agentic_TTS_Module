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
import secrets
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

# 跨区域时按这个顺序兜底 —— _pick_endpoint 之外、用户未配置时也带着。
# Cross-region fallback order.
_FALLBACK_ENDPOINTS = (
    "https://api.minimaxi.com",
    "https://api.minimax.io",
)

# 业务层鉴权/账号错码：换 endpoint 或许能恢复。其它错误（参数、配额、限流）不切。
# Auth/account codes that may be resolved by switching endpoint.
_AUTH_STATUS_CODES = {
    2049,    # invalid api key
    2048,    # auth failed
    1004,    # authentication error
    1002,    # invalid authorization
    1008,    # payment required (key 绑在另一个账户上)
}


_VOICE_NOT_FOUND = 2054


class _VoiceMissing(SynthError):
    """t2a 报 2054：所给 voice_id 在服务端不存在。"""


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
                "MiniMax API key 未配置：在项目根的 .env 里设 TTS_API_KEY=sk-…，"
                "或临时 `export TTS_API_KEY=…` 后再启动服务。"
                "代码与 config.yaml 都不再持有密钥（与 DailyNewsAssistant 一致）。"
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
        static = [
            VoiceInfo(name=n, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      language=lang, description="", speaker=n)
            for n, lang in system_voices
        ]
        # 账号里克隆 / 设计出的音色在 MiniMax 侧就是普通 voice_id，按内置音色用。
        # 不并进来的话 VoiceStore 不认这个名字，GUI 也选不到（docs/issues/006）。
        account = [
            VoiceInfo(name=vid, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      description=desc, speaker=vid)
            for vid, desc in self._account_voices()
        ]
        return static + account

    def _account_voices(self) -> list[tuple[str, str]]:
        """账号音色 (voice_id, 说明)。查不到只记警告——不能因为列表拿不到就起不了服务。"""
        import httpx

        try:
            r = httpx.post(f"{self.base_url}/v1/get_voice",
                           headers={"Authorization": f"Bearer {self._api_key}"},
                           json={"voice_type": "all"}, timeout=10)
            data = _checked(r, "查询账号音色")
        except (httpx.HTTPError, SynthError, ValueError) as exc:
            _logger.warning("拿不到 MiniMax 账号音色，只列静态系统音色：%s", exc)
            return []
        kinds = (("voice_cloning", "克隆音色"), ("voice_generation", "设计音色"))
        return [
            (v["voice_id"], f"{label} · {v.get('created_time', '')}".rstrip(" ·"))
            for key, label in kinds
            for v in (data.get(key) or [])
            if v.get("voice_id")
        ]

    def languages(self) -> list[str]:
        return ["Chinese", "English", "Japanese", "Korean", "auto"]

    # ── 合成 / synthesis ──

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        self.require_mode(chunk.mode)

        if chunk.mode is not Mode.VOICE_CLONE:
            return self._synthesize_with(chunk, chunk.speaker or "male-qn-qingse")

        voice_id = self._resolve_or_clone_voice(chunk)
        try:
            return self._synthesize_with(chunk, voice_id)
        except _VoiceMissing:
            # 缓存里的 voice_id 在服务端已不存在（曾被缓存的失败克隆，或 MiniMax
            # 回收了久未使用的临时音色）。不清掉的话之后每一段都会 2054，永远不自愈。
            _logger.warning("cached voice_id=%s 服务端不存在，作废缓存并重新克隆", voice_id)
            self._forget_voice(voice_id)
            return self._synthesize_with(chunk, self._resolve_or_clone_voice(chunk))

    def _synthesize_with(self, chunk: SynthChunk, voice_id: str) -> RawAudio:
        body = self._build_request(chunk, voice_id)

        # 端点候选：用户配置的优先；其它按白名单顺序作为备用。
        # Endpoint candidates: configured first; then whitelist as fallback.
        # 这样 key 跨域（Token Plan key 配到海外 endpoint、或反之）能自动恢复。
        # Switching on auth errors lets a misrouted key still recover automatically.
        candidates = self._endpoint_candidates()
        attempts_log: list[str] = []

        import httpx
        last_exc: Exception | None = None
        for idx, ep in enumerate(candidates):
            # 同一端点最多 1 次网络层重试（connect/read timeout 等瞬时错误），
            # 再不行就换下一个端点。鉴权错误（见 _AUTH_STATUS_CODES）直接换端点不重试。
            for transport_retry in range(2):
                started = time.perf_counter()
                try:
                    # 连接 5s 与读取超时分开：VPN/DNS 阻塞时不能把读超时也拖到 timeout_s
                    timeouts = httpx.Timeout(connect=5.0, read=self.timeout_s,
                                             write=self.timeout_s, pool=5.0)
                    with httpx.Client(timeout=timeouts) as client:
                        resp = client.post(
                            f"{ep}/v1/t2a_v2",
                            headers={"Authorization": f"Bearer {self._api_key}",
                                     "Content-Type": "application/json"},
                            json=body,
                        )
                    data = resp.json()
                    elapsed = time.perf_counter() - started
                except (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.PoolTimeout, httpx.NetworkError) as exc:
                    last_exc = exc
                    attempts_log.append(f"{ep}/transport:{type(exc).__name__}")
                    _logger.warning("MiniMax transport @%s: %s（第 %d 次）",
                                    ep, exc, transport_retry + 1)
                    if transport_retry == 1:
                        break                       # 该端点两次都失败，跳到下一个端点
                    time.sleep(0.6 * (transport_retry + 1))
                    continue
                except Exception as exc:             # JSON 解析等真·异常，不再重试
                    raise SynthError(f"MiniMax 调用失败：{exc}") from exc

                code = int((data.get("base_resp") or {}).get("status_code", 0) or 0)
                if code == 0:
                    break                           # 成功；跳出 transport_retry 循环
                if code in _AUTH_STATUS_CODES:
                    attempts_log.append(f"{ep}/auth:{code}")
                    _logger.warning("MiniMax auth 失败 @%s: code=%s msg=%s —— 尝试备用端点",
                                    ep, code,
                                    (data.get("base_resp") or {}).get("status_msg", ""))
                    break                           # 直接试下一端点（不耗 transport_retry）
                # 业务错（参数、配额、限流等）—— 不重试也不换端点
                msg = (data.get("base_resp") or {}).get("status_msg", "")
                error = _VoiceMissing if code == _VOICE_NOT_FOUND else SynthError
                raise error(
                    f"MiniMax 业务错（{ep}）：{code} {msg}".rstrip()
                )
            else:
                # transport_retry for 循环正常结束（没 break）= 两次都 transport 失败
                continue
            # 走到这里说明已经 break（成功或换端点）
            if (data.get("base_resp") or {}).get("status_code", 0) == 0:
                break
            # 否则是 auth 失败，继续下一个候选端点
            continue

        else:
            # 所有候选端点都失败
            log = "；".join(attempts_log) or "(无记录)"
            detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "见上方 attempts"
            raise SynthError(
                f"MiniMax 所有端点均失败：{log}；最近一次：{detail}"
            )

        # 走到这里说明拿到 200 且 base_resp.status_code == 0
        elapsed = locals().get("elapsed", 0.0)

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
                "endpoint_used": ep,
            },
            instruct_applied=bool(chunk.instruct),
        )

    def _endpoint_candidates(self) -> list[str]:
        """
        端点候选顺序 / Ordered endpoint candidates.

        1. 用户显式配置（非空）
        2. 白名单按 key 前缀推出的默认（避免重复）
        3. 兜底：另一侧的官方端点
        """
        configured = (self.config.api.endpoint or "").strip()
        keys = [configured] if configured else []

        primary = _pick_endpoint(self._api_key, "")
        if primary not in keys:
            keys.append(primary)

        # 兜底：把另一个域也带上（除非和 primary 相同）
        for ep in _FALLBACK_ENDPOINTS:
            if ep not in keys:
                keys.append(ep)
        return [k.rstrip("/") for k in keys if k]

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
        # 后缀用随机数而非秒级时间戳：2054 重克隆就在同一秒内，时间戳会撞出同一个 id。
        vid = f"v{digest[:8]}_{secrets.token_hex(4)}"

        with httpx.Client(timeout=self.timeout_s) as client:
            # 1) 上传文件
            files = {"file": ("ref.mp3", audio_bytes, "audio/mpeg")}
            data = {"purpose": "voice_clone"}
            r = client.post(
                f"{self.base_url}/v1/files/upload",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files=files, data=data,
            )
            file_id = _checked(r, "上传参考音频")["file"]["file_id"]

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
            # MiniMax 的业务错是 HTTP 200 + base_resp 非 0；只看 HTTP 码会把失败的
            # 克隆当成功写进缓存，之后每次合成都 2054 voice id not exist（实测踩到）。
            _checked(r, "克隆音色")
            return vid

    def _forget_voice(self, voice_id: str) -> None:
        self._cache = {k: v for k, v in self._cache.items()
                       if v.get("voice_id") != voice_id}
        self._save_cache()

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


def _checked(resp, step: str) -> dict:
    """HTTP 码与 base_resp 都过关才返回 JSON，否则抛带 MiniMax 原始错误码的 SynthError。"""
    resp.raise_for_status()
    data = resp.json()
    base = data.get("base_resp") or {}
    code = int(base.get("status_code", 0) or 0)
    if code != 0:
        raise SynthError(f"MiniMax {step}失败：{code} {base.get('status_msg', '')}".rstrip())
    return data


def _to_language_boost(language: str) -> str:
    return {"Chinese": "Chinese", "English": "English",
            "Japanese": "Japanese", "Korean": "Korean"}.get(language, "auto")


def _get_api_key_from_env() -> Optional[str]:
    import os
    return os.environ.get("TTS_API_KEY") or os.environ.get("MINIMAX_API_KEY")


# 引擎注册
__all__ = ["MiniMaxEngine"]