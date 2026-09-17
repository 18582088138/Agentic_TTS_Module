"""
对外类型 / Public types —— 请求用 pydantic（要校验、要当 HTTP body），
结果用 dataclass（里面带 numpy 波形，塞进 pydantic 只会打架）。
Requests are pydantic models; results are dataclasses because they carry numpy audio.

────────────────────────────────────────────────────────────────────────────
两条贯穿全模块的约定
────────────────────────────────────────────────────────────────────────────
1. **能力由引擎静态声明**（`Capability`），不加载权重就能查 ——
   GUI 要在开机时就把不支持的开关置灰，不可能为了查能力先载 4 GB 权重。
   Capabilities are declared statically so the GUI can grey out controls without
   loading any weights.
2. **结果必须自带「实际生效」的参数**。只记"我传了什么"，manifest 里就全是 null，
   而事后最想知道的恰恰是"这条音频到底用什么参数跑出来的"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- 枚举 / enums


class Mode(str, Enum):
    """
    合成模式 / Synthesis mode.

    ⚠️ Qwen3-TTS 三个模式**各绑一个 checkpoint，门禁在模型侧硬写** ——
    换模式就是换权重（8 GB 卡上意味着一次卸载+重载）。
    Each mode is bound to its own checkpoint upstream; switching modes swaps weights.
    """

    CUSTOM_VOICE = "custom_voice"   # 内置音色 / built-in speaker id
    VOICE_DESIGN = "voice_design"   # 自然语言描述音色 / voice from a description
    VOICE_CLONE = "voice_clone"     # 参考音频克隆 / clone from reference audio


class Capability(str, Enum):
    """
    引擎能力 / Engine capability. 静态声明，GUI 据此置灰开关。

    `PAUSE`（段间插静音）**不在这里** —— 它由门面在拼接层实现，对所有引擎都成立。
    """

    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"
    CLONE_ICL = "clone_icl"           # 克隆的 ICL 模式（要 ref_text，音色更像）
    INSTRUCT = "instruct"             # 指令控制语气/语速
    VOCAL_EVENTS = "vocal_events"     # **原生**事件标记；Qwen3 开源权重没有
    BATCH = "batch"                   # 一次调用多条文本


class SplitRule(str, Enum):
    """一整段文本自动拆成多段的规则 / How to auto-split one blob into segments."""

    CHARS = "chars"     # 按字数上限（仍在标点处落刀，不硬切）
    PUNCT = "punct"     # 按句末标点，一句一段
    ROLE = "role"       # 按「角色：台词」前缀
    BLANK = "blank"     # 按空行（作者已经分好段的情况）


# ------------------------------------------------------------- 请求 / requests


class SynthRequest(BaseModel):
    """
    一次合成请求 / One synthesis request.

    `mode` 留空则按参数推断（见 `infer_mode`）；参数与 `mode` 冲突时**直接报错**，
    不做静默忽略。
    `engine` 留空则用 config.engine.backend；显式指定时按此引擎合成（不触发 fallback）。
    """

    text: str = Field(min_length=1)
    mode: Optional[Mode] = None
    engine: Optional[str] = None           # None = 用 config.engine.backend；显式值跳过 fallback
    voice: Optional[str] = None            # 内置音色名，或 voices.yaml 里的档案名
    language: str = "Chinese"
    instruct: Optional[str] = None

    # 克隆参数 / voice-clone parameters
    ref_audio: Optional[str] = None        # 本地路径或 data:audio;base64（不支持 URL，见 docs/00 §2.10）
    ref_text: Optional[str] = None
    x_vector_only: bool = False            # True = 只用 speaker embedding（不需要 ref_text）

    # 逐次覆盖 / per-call overrides
    gen: dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    role: Optional[str] = None             # 多段时的角色名；换角色处用更长的停顿
    pause_ms: Optional[int] = None         # 合并时本段之后的静音；None = 用配置

    # 文本链路开关，None = 跟随配置 / text-pipeline switches; None follows config
    prepare_text: Optional[bool] = None
    segment: Optional[bool] = None
    events: Optional[bool] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "SynthRequest":
        mode = self.mode or infer_mode(self)
        if mode is Mode.VOICE_CLONE and not self.ref_audio:
            raise ValueError("voice_clone 需要 ref_audio")
        if mode is not Mode.VOICE_CLONE and self.ref_audio:
            raise ValueError(f"mode={mode.value} 不该带 ref_audio（克隆请用 mode=voice_clone）")
        if mode is Mode.VOICE_DESIGN and not self.instruct:
            raise ValueError("voice_design 需要 instruct 描述目标音色")
        if mode is Mode.VOICE_CLONE and not self.x_vector_only and not self.ref_text:
            # ICL 模式上游硬要求 ref_text；这里先拦下来，省掉一次几十秒的加载
            raise ValueError("ICL 克隆需要 ref_text（或设 x_vector_only=true 走纯 x-vector）")
        return self

    def resolved_mode(self) -> Mode:
        """最终生效的模式 / The mode that will actually be used."""
        return self.mode or infer_mode(self)


class BatchSynthRequest(BaseModel):
    """
    多段合成 / Multi-segment synthesis.

    每段可以各带自己的音色/模式/指令/事件 —— GUI 的多段面板就是这个结构。
    `merge=True` 时额外产出一条拼接好的整段音频。
    """

    segments: list[SynthRequest] = Field(min_length=1)
    merge: bool = True
    save_segments: bool = True         # 每段单独落盘 / write each segment to its own file
    name: Optional[str] = None         # 输出子目录名；None = 时间戳


class SplitRequest(BaseModel):
    """把一整段文本拆成多段 / Split one blob of text into segments."""

    text: str = Field(min_length=1)
    rule: SplitRule = SplitRule.PUNCT
    max_chars: int = 60
    # 0 = 不合并碎句（"一句一段"就是一句一段）；调大可把「好。」这类并掉
    min_chars: int = 0
    role_voices: dict[str, str] = Field(default_factory=dict)   # 角色名 → 音色名


# --------------------------------------------------------------- 结果 / results


@dataclass
class RawAudio:
    """
    引擎返回的原始音频 / What an engine hands back for one chunk.

    引擎只负责这一层：一段文本 → 一段波形 + 实际生效的参数。
    分段、拼接、失控兜底、文本预处理都在门面上，对所有引擎共享。
    """

    wav: np.ndarray
    sample_rate: int
    effective_gen: dict[str, Any] = field(default_factory=dict)
    instruct_applied: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.wav) / self.sample_rate if self.sample_rate else 0.0


@dataclass
class SegmentResult:
    """一段的结果 / The result for one segment."""

    index: int
    text: str                     # 原文 / as requested
    spoken_text: str              # 实际送进模型的文本（预处理+去事件标记后）
    mode: Mode
    voice: Optional[str]
    role: Optional[str]
    seconds: float                # **音频时长**，不是耗时
    sample_rate: int
    rtf: float
    seed: int
    elapsed: float = 0.0          # **墙钟耗时**（秒）：这一段实际花了多久
    chunks: int = 1               # 这一段内部被切成几块合成 / internal chunk count
    retries: int = 0
    failed: bool = False
    effective_gen: dict[str, Any] = field(default_factory=dict)
    instruct: Optional[str] = None
    instruct_applied: bool = False
    warnings: list[str] = field(default_factory=list)
    path: Optional[str] = None
    wav: Optional[np.ndarray] = None      # 不进 JSON / never serialised

    def as_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "wav"}
        data["mode"] = self.mode.value
        return data


@dataclass
class SynthResult:
    """
    一次合成（可能含多段）的完整结果 / The full result, possibly multi-segment.

    可观测字段是刻意冗余的：`rtf`、`effective_gen`、`retries`、`warnings`
    合起来才能回答「这条音频是怎么来的」。
    """

    segments: list[SegmentResult]
    sample_rate: int
    seconds: float                        # **音频时长**，不是耗时
    rtf: float
    elapsed: float = 0.0                  # **墙钟耗时**（秒）：这次调用实际花了多久
    engine: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    path: Optional[str] = None            # 合并后的整段音频 / the merged file
    subtitle_path: Optional[str] = None   # 合并音频对应的字幕 / the subtitle file
    output_dir: Optional[str] = None
    wav: Optional[np.ndarray] = None      # 不进 JSON

    @property
    def ok(self) -> bool:
        return bool(self.segments) and not any(s.failed for s in self.segments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sample_rate": self.sample_rate,
            "seconds": round(self.seconds, 3),
            "elapsed": round(self.elapsed, 3),
            "rtf": round(self.rtf, 3),
            "path": self.path,
            "subtitle_path": self.subtitle_path,
            "output_dir": self.output_dir,
            "engine": self.engine,
            "warnings": self.warnings,
            "segments": [s.as_dict() for s in self.segments],
        }


class VoiceInfo(BaseModel):
    """一个可用音色 / One selectable voice —— 内置 speaker 或本地档案。"""

    name: str
    mode: Mode
    engine: str
    language: Optional[str] = None
    description: Optional[str] = None
    builtin: bool = True
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False


# ------------------------------------------------------------- 推断 / inference


def infer_mode(req: "SynthRequest") -> Mode:
    """
    从参数推断模式 / Infer the mode from the parameters.

    规则写死在一处并有单测 —— 三个 API 各绑一个 checkpoint，推断错就是加载错权重，
    代价是几十秒。
    Rules live in exactly one place, because guessing wrong loads the wrong 4 GB.

    优先级 / Precedence:
        1. `ref_audio` 有值 → `voice_clone`
        2. `voice` 为空且 `instruct` 有值 → `voice_design`
        3. 其他 → `custom_voice`
    """
    if req.ref_audio:
        return Mode.VOICE_CLONE
    if not req.voice and req.instruct:
        return Mode.VOICE_DESIGN
    return Mode.CUSTOM_VOICE


__all__ = [
    "BatchSynthRequest",
    "Capability",
    "Mode",
    "RawAudio",
    "SegmentResult",
    "SplitRequest",
    "SplitRule",
    "SynthRequest",
    "SynthResult",
    "VoiceInfo",
    "infer_mode",
]
