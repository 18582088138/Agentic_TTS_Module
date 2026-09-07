"""
Qwen3-TTS 引擎（torch，cpu/cuda）/ The Qwen3-TTS engine on PyTorch.

────────────────────────────────────────────────────────────────────────────
三件必须知道的事实（全部实测/读源码核实，见 docs/00_research.md）
────────────────────────────────────────────────────────────────────────────
1. **三个 API 各绑一个 checkpoint，门禁在模型侧硬写。** 换模式 = 换权重。
   custom_voice → *-CustomVoice　voice_design → *-VoiceDesign　voice_clone → *-Base
2. **8 GB 卡装得下一个 1.7B、装不下两个**（bf16 权重 3.9 GB，跑起来约 5 GB）。
   所以 `ModelPool` 默认 `max_resident=1`，**轮动加载**：换 checkpoint 先卸再载，
   且必须显式 `empty_cache()` —— 仅 `del` 不会把显存还给驱动，
   下一次加载照样 OOM，而报错只说「显存不足」，完全看不出是上一个模型赖着没走。
3. **0.6B CustomVoice 会静默丢弃 `instruct`**（上游 `qwen3_tts_model.py:799`）。
   我们把「instruct 有没有真的进去」如实记进结果，免得听不出语气变化却查不出原因。

⚠️ torch 一律在**函数体内**import：GUI 要在不加载 torch 的前提下查能力表。
torch is imported inside functions so the capability lookup stays torch-free.
"""

from __future__ import annotations

import base64
import gc
import importlib.metadata
import platform
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agentic_tts.core.config import Config, ensure_qwen_tts_importable
from agentic_tts.core.errors import EngineError, SynthError
from agentic_tts.core.logging import get_logger
from agentic_tts.core.registry import register_engine
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo
from agentic_tts.engines.base import SynthChunk, TTSEngine

_logger = get_logger("engine.qwen3")

# mode → checkpoint 种类（配置里的键名）
_CHECKPOINT_FOR_MODE = {
    Mode.CUSTOM_VOICE: "custom_voice",
    Mode.VOICE_DESIGN: "voice_design",
    Mode.VOICE_CLONE: "base",
}

# `generate_*` 的具名参数，不属于 HF `generate()` 的 kwargs，要单独拆出来传
_STANDALONE = ("non_streaming_mode",)

# 一个 1.7B checkpoint 的实测占用（参数 1.917 B + 声码器 171 M）
_WEIGHT_GB = {"bfloat16": 3.9, "float16": 3.9, "float32": 7.8}
# 加上 KV cache、激活与分配器碎片之后的经验值
_RUNTIME_GB = {"bfloat16": 5.0, "float16": 5.0, "float32": 9.5}

# 内置音色静态表（来自官方 README）。**不加载权重就能给出** —— GUI 要用。
# 模型载入后会用 `get_supported_speakers()` 刷新一遍。
_BUILTIN_SPEAKERS = [
    ("Vivian", "Chinese", "明亮、略带锋利的年轻女声"),
    ("Serena", "Chinese", "温暖、柔和的年轻女声"),
    ("Uncle_Fu", "Chinese", "低沉醇厚的成熟男声"),
    ("Dylan", "Chinese", "京片子男声，音色清亮自然（方言只能靠音色拿到）"),
    ("Eric", "Chinese", "成都话男声，略带沙哑的明亮感"),
    ("Ryan", "English", "节奏感强的动感男声"),
    ("Aiden", "English", "阳光的美式男声，中频清晰"),
    ("Ono_Anna", "Japanese", "俏皮轻盈的日语女声"),
    ("Sohee", "Korean", "情感丰富的温暖韩语女声"),
]

_LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German",
              "French", "Russian", "Portuguese", "Spanish", "Italian"]


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_snapshot() -> dict[str, str]:
    """
    环境快照 / Environment snapshot.

    版本号是**事后复现的唯一线索**：同一份输入在 transformers 4.57 与 4.58 下
    可能给出不同音频，没记版本就无从解释。
    """
    return {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "numpy": _version("numpy"),
        "soundfile": _version("soundfile"),
        "librosa": _version("librosa"),
    }


def free_memory() -> None:
    """
    释放已卸载模型占的内存与显存 / Release memory held by unloaded models.

    ⚠️ **仅 `del` 是不够的**：CUDA 的缓存分配器不会把显存还给驱动。
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def vram_snapshot() -> Optional[dict[str, float]]:
    """当前显存占用；CPU 上返回 None / Current VRAM usage, None on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        }
    except ImportError:
        return None


def estimate_vram_gb(dtype: str, resident: int = 1) -> float:
    """跑起来要多少显存 / VRAM needed to actually run（含 KV cache 与碎片）。"""
    return _RUNTIME_GB.get(dtype, _RUNTIME_GB["float32"]) * max(resident, 1)


def resolve_device(raw: str) -> str:
    """
    设备解析 / Resolve the device string.

    `auto` → 有 CUDA 就 `cuda:0`，否则 `cpu`。写死配置里的 `cuda` 在没有卡的机器上
    要给出**能照着改的**报错，而不是 torch 那句难懂的 assert。
    """
    want = (raw or "auto").strip().lower()
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError as exc:  # pragma: no cover
        raise EngineError(f"需要 torch 才能解析设备：{exc}") from exc

    if want == "auto":
        return "cuda:0" if has_cuda else "cpu"
    if want.startswith("cuda") and not has_cuda:
        raise EngineError(
            "配置要求 cuda，但 torch 看不到 CUDA 设备　"
            f"（torch={_version('torch')}，多半装的是 +cpu 版）　"
            "改 engine.device 为 cpu 或 auto，或装 CUDA 版 torch（见 docs/05_deployment.md）")
    return "cuda:0" if want == "cuda" else want


def resolve_dtype_name(raw: str, device: str) -> str:
    """
    dtype 解析 / Resolve the dtype name.

    `auto`：CPU 上 float32（cpu 的 bf16 算子慢且不全），CUDA 上 bfloat16
    （8 GB 卡上 float32 的 9.5 GB 是直接 OOM）。
    """
    want = (raw or "auto").strip().lower()
    if want != "auto":
        return want
    return "bfloat16" if device.startswith("cuda") else "float32"


class _Loaded:
    """一个已加载的 checkpoint / One loaded checkpoint."""

    def __init__(self, model: Any, kind: str, path: Path, device: str, dtype: str,
                 load_seconds: float) -> None:
        self.model = model
        self.kind = kind
        self.path = path
        self.device = device
        self.dtype = dtype
        self.load_seconds = load_seconds
        config = getattr(getattr(model, "model", None), "config", None)
        self.model_type = str(getattr(config, "tts_model_type", "") or "")
        self.model_size = str(getattr(config, "tts_model_size", "") or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint": self.path.name,
            "checkpoint_path": str(self.path),
            "tts_model_type": self.model_type,
            "tts_model_size": self.model_size,
            "device": self.device,
            "dtype": self.dtype,
            "load_seconds": round(self.load_seconds, 2),
        }


@register_engine("qwen3")
@register_engine("qwen3_tts")
class Qwen3TTSEngine(TTSEngine):
    """Qwen3-TTS 1.7B（torch cpu/cuda），支持内置音色/音色设计/音色克隆。"""

    name = "qwen3"
    # 12Hz tokenizer 实测 12.5 codes/秒 —— 失控判据靠它把 max_new_tokens 换成秒
    codes_per_second = 12.5
    default_max_new_tokens = 8192

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device = ""
        self.dtype = ""
        # OrderedDict 做 LRU：最近用过的移到末尾，轮动时弹出最前面的
        self._pool: "OrderedDict[str, _Loaded]" = OrderedDict()
        self._speakers: Optional[list[str]] = None
        self._languages: Optional[list[str]] = None

    # ---------------------------------------------------------------- 能力

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        # ⚠️ **没有** VOCAL_EVENTS：开源权重里不存在原生事件 token
        # （tokenizer 特殊符号全表核对 + 全仓库 grep 零命中，见 docs/00_research.md §3）。
        return {
            Capability.CUSTOM_VOICE,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.INSTRUCT,
            Capability.BATCH,
        }

    def builtin_voices(self) -> list[VoiceInfo]:
        """
        内置音色 / Built-in speakers.

        静态表打底（不加载权重也能给出，GUI 要用）；模型载入过就用模型的实际列表过滤，
        免得静态表与权重不一致时给出一个模型不认的名字。
        """
        names = set(self._speakers) if self._speakers else None
        return [
            VoiceInfo(name=name, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      language=lang, description=desc, speaker=name)
            for name, lang, desc in _BUILTIN_SPEAKERS
            if names is None or name in names
        ]

    def languages(self) -> list[str]:
        """
        支持的语言 / Supported languages.

        ⚠️ 方言**不在**这里：上游把 `*_dialect` 用 `if "dialect" not in language_id`
        过滤掉了，方言只能通过音色拿到（Dylan 京片子 / Eric 川普）。
        """
        return list(self._languages or _LANGUAGES)

    # ------------------------------------------------------- 轮动加载 / rotation

    def _resolve_runtime(self) -> None:
        if not self.device:
            self.device = resolve_device(self.config.engine.device)
            self.dtype = resolve_dtype_name(self.config.engine.dtype, self.device)

    def _check_vram_headroom(self) -> None:
        """
        加载前预检显存 / Check VRAM headroom before loading.

        加载一个 1.7B 要几十秒；撞到 OOM 才发现 dtype 配错，那几十秒就白等了，
        而 CUDA 的 OOM 报错也不会提示「换成 bfloat16 就行」。
        """
        if not self.config.engine.vram_guard:
            return
        snapshot = vram_snapshot()
        if snapshot is None:
            return  # CPU：内存不足会走系统 swap，不在这里拦

        need = estimate_vram_gb(self.dtype)
        free = snapshot["total_gb"] - snapshot["allocated_gb"]
        if free >= need:
            return

        if self.dtype == "float32":
            hint = (f"　把 engine.dtype 改成 bfloat16（约需 "
                    f"{estimate_vram_gb('bfloat16'):.1f} GB，float32 要 {need:.1f} GB）")
        elif snapshot["allocated_gb"] > 0.5:
            hint = (f"　显存里还占着 {snapshot['allocated_gb']:.1f} GB —— "
                    "上一个 checkpoint 没卸干净？engine.max_resident 应为 1")
        else:
            hint = "　这张卡装不下本模型，换机器或换更小的 checkpoint"
        raise EngineError(
            f"显存不够：需要约 {need:.1f} GB，可用 {free:.1f} GB"
            f"（总 {snapshot['total_gb']:.1f} GB）。{hint}")

    def _get(self, kind: str) -> _Loaded:
        """
        取一个已加载的 checkpoint，没有就加载 / Get a checkpoint, loading if needed.

        **顺序很重要：先卸后载。** 反过来会让两个模型同时在显存里，
        恰好是 8 GB 卡上要避免的那件事。
        Evict before loading; the reverse puts both checkpoints in VRAM at once.
        """
        if kind in self._pool:
            self._pool.move_to_end(kind)
            return self._pool[kind]

        self._resolve_runtime()
        while len(self._pool) >= self.config.engine.max_resident:
            evicted, _ = self._pool.popitem(last=False)
            _logger.info("轮动加载：卸载 %s 腾出显存", evicted)
            free_memory()

        self._pool[kind] = self._load(kind)
        return self._pool[kind]

    def _load(self, kind: str) -> _Loaded:
        path = self.config.checkpoint_path(kind)
        if not path.is_dir():
            raise EngineError(
                f"checkpoint 目录不存在：{path}　"
                f"modelscope download --model Qwen/{path.name} --local_dir {path}")

        ensure_qwen_tts_importable(self.config.repo_path)
        self._check_vram_headroom()

        import torch
        from qwen_tts import Qwen3TTSModel

        kwargs: dict[str, Any] = {
            "device_map": self.device,
            "dtype": getattr(torch, self.dtype),
        }
        if self.config.engine.attn_impl:
            kwargs["attn_implementation"] = self.config.engine.attn_impl

        _logger.info("加载 %s（%s，%s，%s）…", path.name, self.device, self.dtype,
                     self.config.engine.attn_impl)
        started = time.perf_counter()
        try:
            model = Qwen3TTSModel.from_pretrained(str(path), **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"加载 {path.name} 失败：{type(exc).__name__}: {exc}　"
                f"（device={self.device} dtype={self.dtype} "
                f"attn={self.config.engine.attn_impl}）") from exc
        elapsed = time.perf_counter() - started
        _logger.info("加载完成，耗时 %.1f 秒", elapsed)

        loaded = _Loaded(model, kind, path, self.device, self.dtype, elapsed)
        self._refresh_metadata(model)
        return loaded

    def _refresh_metadata(self, model: Any) -> None:
        """模型载入后刷新音色/语言表 / Refresh voice and language tables after load."""
        for attr, target in (("get_supported_speakers", "_speakers"),
                             ("get_supported_languages", "_languages")):
            getter = getattr(model, attr, None)
            if not callable(getter):
                continue
            try:
                value = getter()
            except Exception:  # noqa: BLE001 - 元信息拿不到不该让合成失败
                continue
            if value:
                setattr(self, target, sorted(str(v) for v in value))

    def release(self) -> None:
        if self._pool:
            _logger.info("释放 %s", "、".join(self._pool))
        self._pool.clear()
        free_memory()

    @property
    def resident(self) -> list[str]:
        """当前驻留的 checkpoint，按最近使用排序 / Currently resident, LRU order."""
        return list(self._pool)

    # ------------------------------------------------------------ 合成 / synth

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        self.require_mode(chunk.mode)
        loaded = self._get(_CHECKPOINT_FOR_MODE[chunk.mode])
        self._seed(chunk.seed)

        overrides = dict(chunk.gen)
        # 私有测试钩子不该漏给上游 / private test hooks never reach upstream
        for key in [k for k in overrides if k.startswith("_")]:
            overrides.pop(key)
        standalone = {k: overrides.pop(k) for k in _STANDALONE if k in overrides}
        effective = self._effective_gen(loaded.model, overrides)

        warnings: list[str] = []
        instruct = chunk.instruct or None
        instruct_applied = bool(instruct)
        if instruct and loaded.model_size and "0b6" in loaded.model_size:
            # 上游 0.6B 会静默丢弃 instruct。静默正是这个坑的形状，所以我们说出来。
            instruct_applied = False
            warnings.append("0.6B checkpoint 不支持 instruct，上游会静默丢弃（已如实记录）")

        started = time.perf_counter()
        try:
            wavs, sample_rate = self._dispatch(loaded, chunk, instruct, overrides, standalone)
        except Exception as exc:  # noqa: BLE001
            raise SynthError(f"生成失败：{type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started

        if not wavs:
            raise SynthError("模型没有返回任何音频")

        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return RawAudio(
            wav=wav,
            sample_rate=int(sample_rate),
            effective_gen={**effective, **standalone, "generate_seconds": round(elapsed, 2)},
            instruct_applied=instruct_applied,
            warnings=warnings,
        )

    def _dispatch(self, loaded: _Loaded, chunk: SynthChunk, instruct: Optional[str],
                  overrides: dict[str, Any], standalone: dict[str, Any]) -> tuple[list, int]:
        model = loaded.model
        if chunk.mode is Mode.CUSTOM_VOICE:
            return model.generate_custom_voice(
                text=chunk.text,
                speaker=chunk.speaker,
                language=chunk.language,
                instruct=instruct,
                **standalone,
                **overrides,
            )
        if chunk.mode is Mode.VOICE_DESIGN:
            return model.generate_voice_design(
                text=chunk.text,
                instruct=instruct,
                language=chunk.language,
                **standalone,
                **overrides,
            )
        return model.generate_voice_clone(
            text=chunk.text,
            language=chunk.language,
            ref_audio=self._resolve_ref_audio(chunk.ref_audio),
            ref_text=chunk.ref_text,
            x_vector_only_mode=chunk.x_vector_only,
            **standalone,
            **overrides,
        )

    @staticmethod
    def _resolve_ref_audio(ref_audio: Optional[str]) -> Optional[str]:
        """
        解析参考音频 / Resolve the reference audio.

        只接受**本地文件**与 base64。⚠️ 上游的 URL 加载走
        `urllib.request.urlopen`，**不读代理配置**，公司网络下多半超时 ——
        与其等它超时，不如在这里直接拒绝并说清原因。
        URLs are rejected: upstream's loader ignores proxy settings and hangs.
        """
        if not ref_audio:
            return None
        if ref_audio.startswith(("http://", "https://")):
            raise SynthError(
                "ref_audio 不支持 URL（上游加载不读代理配置，公司网络下会超时）　"
                "先下载成本地文件再传路径")
        if ref_audio.startswith("data:audio"):
            header, _, payload = ref_audio.partition(",")
            suffix = ".wav" if "wav" in header else ".mp3"
            handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            handle.write(base64.b64decode(payload))
            handle.close()
            return handle.name
        path = Path(ref_audio)
        if not path.is_file():
            raise SynthError(f"参考音频不存在：{path}")
        return str(path)

    @staticmethod
    def _effective_gen(model: Any, overrides: dict[str, Any]) -> dict[str, Any]:
        """
        算出**实际生效**的生成参数 / Compute the parameters that will take effect.

        合并规则是三级的：**显式传参 > `generation_config.json` > 代码硬编码兜底**。
        只记我们传进去的那几项，结果里就是一堆 null，而事后最想知道的恰恰是
        「这条音频到底用 temperature 0.9 还是 0.7 跑的」。所以直接调模型自己的
        `_merge_generate_kwargs` —— 它就是模型待会儿真正要用的那份。
        """
        merge = getattr(model, "_merge_generate_kwargs", None)
        if not callable(merge):
            _logger.warning("模型没有 _merge_generate_kwargs，只能记传入值（上游改名了？）")
            return {"__merge_unavailable__": True, **overrides}
        try:
            return dict(merge(**overrides))
        except Exception:  # noqa: BLE001
            return {"__merge_failed__": True, **overrides}

    @staticmethod
    def _seed(seed: int) -> None:
        """
        固定随机种子 / Fix the random seed.

        ⚠️ 只固定种子**不足以**让两次完全一致：默认 `do_sample=True` 且主 talker 与
        sub-talker 都在采样。但这已经足够让「换种子重试」确实换出不同结果。
        **不要为了可复现去关采样** —— 实测贪心解码 12 条里 7 条失控，
        官方采样默认值 0 条失控。
        """
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------ 元信息 / info

    def info(self) -> dict[str, Any]:
        self._resolve_runtime()
        return {
            **super().info(),
            "device": self.device,
            "dtype": self.dtype,
            "attn_impl": self.config.engine.attn_impl,
            "max_resident": self.config.engine.max_resident,
            "repo_dir": str(self.config.repo_path),
            "models_dir": str(self.config.models_path),
            "checkpoints": {k: str(self.config.checkpoint_path(k))
                            for k in self.config.engine.checkpoints},
            "resident": [item.as_dict() for item in self._pool.values()],
            "vram": vram_snapshot(),
            "environment": environment_snapshot(),
        }


__all__ = [
    "Qwen3TTSEngine",
    "environment_snapshot",
    "estimate_vram_gb",
    "free_memory",
    "resolve_device",
    "resolve_dtype_name",
    "vram_snapshot",
]
