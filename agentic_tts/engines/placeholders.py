"""
未接入的引擎占位 / Not-yet-integrated engines —— Breeze-TTS 2、IndexTTS-2。

────────────────────────────────────────────────────────────────────────────
为什么要留占位而不是干脆不写
────────────────────────────────────────────────────────────────────────────
GUI 要在**开机时**把不支持的开关置灰，也就是说它必须能查到「这个引擎支持哪些能力」，
而这件事不该需要先把引擎跑起来。占位类只声明能力表并在实例化时报清晰的错误 ——
于是配置里写 `backend: index2` 得到的是一句能照着做的话，而不是 ImportError 或
"不认识的引擎"。
The GUI needs the capability table at startup; these classes provide it and fail
with an actionable message on instantiation.

⚠️ 能力表按各自的公开文档填写，**本阶段一条都没有实测** ——
真正接入时必须逐条核对（尤其 `VOCAL_EVENTS`：Qwen3 开源权重是没有的，
这两家宣称有，但没验证之前不算结论）。
"""

from __future__ import annotations

from typing import Any

from agentic_tts.core.errors import CapabilityError
from agentic_tts.core.registry import register_engine
from agentic_tts.core.types import Capability, RawAudio
from agentic_tts.engines.base import SynthChunk, TTSEngine


class _NotImplementedEngine(TTSEngine):
    implemented = False
    upstream = ""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        raise CapabilityError(
            f"引擎 {self.name!r} 本阶段未接入（上游：{self.upstream}）　"
            "可用引擎：qwen3（默认）、sine（离线自测）　"
            "接入方式见 docs/06_add_engine.md —— 只需实现 synthesize() 与能力表")

    def synthesize(self, chunk: SynthChunk) -> RawAudio:  # pragma: no cover
        raise CapabilityError(f"引擎 {self.name!r} 未接入")


@register_engine("breeze2")
class Breeze2Engine(_NotImplementedEngine):
    """Breeze-TTS 2（未接入，能力表按公开文档填写，未实测）。"""

    name = "breeze2"
    upstream = "MediaTek Research Breeze-TTS 2"

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.CUSTOM_VOICE,
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.VOCAL_EVENTS,   # ⚠️ 未实测
        }


@register_engine("index2")
class IndexTTS2Engine(_NotImplementedEngine):
    """IndexTTS-2（未接入，能力表按公开文档填写，未实测）。"""

    name = "index2"
    upstream = "bilibili IndexTTS-2"

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.INSTRUCT,
            Capability.VOCAL_EVENTS,   # ⚠️ 未实测
        }


__all__ = ["Breeze2Engine", "IndexTTS2Engine"]
