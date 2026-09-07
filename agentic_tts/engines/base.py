"""
引擎抽象 / Engine abstraction —— **一个引擎只需要会做一件事：一段文本 → 一段波形。**

────────────────────────────────────────────────────────────────────────────
为什么抽象面要这么窄
────────────────────────────────────────────────────────────────────────────
文本规范化、分段、段间插静音、失控兜底这四件事**与引擎无关**。一旦下沉到各引擎
里实现，就会出现 N 份互相偷偷不一致的分段规则 —— 长文案上给出不同切法，
调试时看不出是哪一份在起作用（DailyNewsAssistant 里 `segment.py` 与 textkit
双份并存正是这个坑）。所以它们全部留在门面 `TTSModule` 上，对所有引擎共享。
Those four concerns live in the facade so every engine shares one implementation.

于是新增一个引擎（Breeze-TTS 2 / IndexTTS-2）的成本是：
实现 `synthesize()` + 声明 `declared_capabilities()`，其余全部免费继承。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from agentic_tts.core.errors import CapabilityError
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo

if TYPE_CHECKING:  # pragma: no cover
    from agentic_tts.core.config import Config


@dataclass
class SynthChunk:
    """
    门面交给引擎的一段活 / One unit of work handed to an engine.

    到这一步文本**已经**预处理过、分好段、事件标记已剥离并折进 `instruct` ——
    引擎拿到的是「可以直接送模型的文本」。
    By this point the text is prepared, segmented, and event-free.
    """

    text: str
    mode: Mode
    language: str = "Chinese"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    gen: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


class TTSEngine(ABC):
    """一个 TTS 引擎 / One TTS engine. 权重一律懒加载 —— 造实例不该花几十秒。"""

    name: str = ""
    implemented: bool = True
    # 声学 token 的帧率，用于把 `max_new_tokens` 换算成时长上限（失控判据）。
    # Qwen3-TTS 12Hz tokenizer 实测 12.5 codes/秒。None = 无法换算，只用启发式判据。
    codes_per_second: Optional[float] = None
    default_max_new_tokens: Optional[int] = None

    def __init__(self, config: "Config") -> None:
        self.config = config

    # -- 能力 / capabilities -------------------------------------------------

    @classmethod
    @abstractmethod
    def declared_capabilities(cls) -> set[Capability]:
        """
        静态能力声明 / Static capability declaration.

        **必须不加载权重就能回答** —— GUI 开机时据此把不支持的开关置灰，
        不可能为了查能力先载几个 GB。
        Must be answerable without loading weights; the GUI greys out controls with it.
        """

    def capabilities(self) -> set[Capability]:
        """实例能力（默认等于静态声明）/ Instance capabilities."""
        return self.declared_capabilities()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def require(self, capability: Capability, hint: str = "") -> None:
        """
        不支持就报错，**绝不静默降级** / Fail loudly instead of degrading silently.

        抛出 / Raises:
            CapabilityError: 当前引擎不支持该能力。
        """
        if not self.supports(capability):
            supported = "、".join(sorted(c.value for c in self.capabilities())) or "（无）"
            raise CapabilityError(
                f"引擎 {self.name!r} 不支持 {capability.value}　已支持：{supported}"
                + (f"　{hint}" if hint else ""))

    def require_mode(self, mode: Mode) -> None:
        """模式 → 能力的映射检查 / Check the capability behind a mode."""
        self.require({
            Mode.CUSTOM_VOICE: Capability.CUSTOM_VOICE,
            Mode.VOICE_DESIGN: Capability.VOICE_DESIGN,
            Mode.VOICE_CLONE: Capability.VOICE_CLONE,
        }[mode])

    # -- 合成 / synthesis ----------------------------------------------------

    @abstractmethod
    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        """
        合成一段 / Synthesise one chunk.

        参数 / Args:
            chunk: 已预处理、已分段的一段活 / a prepared, segmented unit of work.

        返回 / Returns:
            `RawAudio`，其中 `effective_gen` 要尽量填**实际生效**的参数
            （而不是传入值）—— 事后最想知道的就是这个。

        抛出 / Raises:
            EngineError: 加载失败。
            SynthError: 生成失败。
        """

    # -- 元信息 / metadata ---------------------------------------------------

    def builtin_voices(self) -> list[VoiceInfo]:
        """
        引擎内置音色 / Voices baked into the engine.

        必须在**不加载权重**的前提下给出（静态表打底）；模型加载后可以刷新。
        """
        return []

    def languages(self) -> list[str]:
        """支持的语言 / Supported languages（静态表打底，同上）。"""
        return []

    def duration_cap_seconds(self, gen: dict[str, Any] | None = None) -> Optional[float]:
        """
        这次生成的时长硬上限 / The hard duration ceiling for this generation.

        失控的**主判据**是「时长撞上限」：模型停不下来时会一直生成到
        `max_new_tokens`，产出几十秒噪声且不报错。
        返回 None 表示无法换算，此时只能用启发式判据（字数 → 预期时长）。
        """
        if not self.codes_per_second:
            return None
        tokens = (gen or {}).get("max_new_tokens") or self.default_max_new_tokens
        if not tokens:
            return None
        return float(tokens) / self.codes_per_second

    def info(self) -> dict[str, Any]:
        """身份快照，进 SynthResult / An identity snapshot for the result."""
        return {
            "engine": self.name,
            "capabilities": sorted(c.value for c in self.capabilities()),
        }

    def release(self) -> None:
        """卸载权重、释放显存 / Unload weights and free memory."""

    def __enter__(self) -> "TTSEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


__all__ = ["SynthChunk", "TTSEngine"]
