"""
离线自测引擎 / Offline self-test engine —— 只产正弦波，零重依赖。

────────────────────────────────────────────────────────────────────────────
为什么值得为它写一个引擎
────────────────────────────────────────────────────────────────────────────
门面、文本链路、HTTP 服务、GUI 这四层的正确性**与神经网络无关**，但如果它们的
测试都要加载 4 GB 权重，一次跑几十秒起、还得有显存，结果就是没人跑测试。
有了它，S4–S6 的测试全部秒级离线，真实模型只在引擎自己的用例里跑。
With it, the facade / text / server / GUI tests run in milliseconds and need no weights.

音频是确定性的：同样的 (文本, 音色, 种子) 出同样的波形，所以可以断言。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentic_tts.core.registry import register_engine
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo
from agentic_tts.engines.base import SynthChunk, TTSEngine

# 说话速度：中文约 5 字/秒。用它把文本长度换成时长，好让分段/拼接/失控逻辑有真实比例。
_CHARS_PER_SEC = 5.0
_SAMPLE_RATE = 24000


@register_engine("sine")
class SineEngine(TTSEngine):
    """正弦波占位引擎（离线自测用，不是真 TTS）/ Sine-wave stub for offline tests."""

    name = "sine"
    codes_per_second = 12.5
    # 刻意设得很小：失控用例要造出「撞上限」的音频，8192 tokens 换算成 655 秒
    # 会让测试产生 60 MB 波形。128 → 上限 10.24 秒，够用且便宜。
    default_max_new_tokens = 128

    _VOICES = {
        "test_female": (330.0, "Chinese"),
        "test_male": (165.0, "Chinese"),
        "test_en": (220.0, "English"),
    }

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # 按**文本**计数，而不是按调用次数：引擎实例在整个进程里复用，
        # 用全局计数会让"前 N 次失控"这个钩子被之前的调用消耗掉（已踩到）。
        self._seen: dict[str, int] = {}

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        # 三种模式与 instruct 都"支持"（只是不影响音色），但**不声明** VOCAL_EVENTS ——
        # 门面因此会走「事件转 instruct 并加 warning」那条路，正好覆盖到那段逻辑。
        return {
            Capability.CUSTOM_VOICE,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.CLONE_ICL,
            Capability.INSTRUCT,
            Capability.BATCH,
        }

    def builtin_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(name=name, mode=Mode.CUSTOM_VOICE, engine=self.name,
                      language=lang, speaker=name, description=f"{freq:.0f} Hz 正弦波")
            for name, (freq, lang) in self._VOICES.items()
        ]

    def languages(self) -> list[str]:
        return ["Chinese", "English"]

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        """
        产一段正弦波 / Produce one sine tone.

        识别两个测试用的私有参数 / Two private test hooks:
            `_force_seconds`: 直接指定时长（秒）。
            `_simulate_runaway`: 同一段文本的前 N 次返回「撞上限」的音频，验失控重试。
        """
        gen = dict(chunk.gen)
        runaway_calls = int(gen.pop("_simulate_runaway", 0) or 0)
        forced = gen.pop("_force_seconds", None)

        seen = self._seen.get(chunk.text, 0) + 1
        self._seen[chunk.text] = seen
        cap = self.duration_cap_seconds(gen) or 10.0
        if seen <= runaway_calls:
            seconds = cap * 0.99          # 撞上限 ⇒ 门面应判失控并换种子重试
        elif forced is not None:
            seconds = float(forced)
        else:
            seconds = max(len(chunk.text) / _CHARS_PER_SEC, 0.2)

        freq, _ = self._VOICES.get(chunk.speaker or "", (440.0, "Chinese"))
        # 种子只改相位：这样「换种子重试」确实换出不同波形，但时长不变，便于断言
        phase = (chunk.seed % 360) * np.pi / 180.0
        t = np.arange(int(seconds * _SAMPLE_RATE), dtype=np.float32) / _SAMPLE_RATE
        wav = (0.2 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)

        return RawAudio(
            wav=wav,
            sample_rate=_SAMPLE_RATE,
            effective_gen={"max_new_tokens": self.default_max_new_tokens, **gen},
            instruct_applied=bool(chunk.instruct),
        )

    def info(self) -> dict[str, Any]:
        return {**super().info(), "sample_rate": _SAMPLE_RATE, "note": "正弦波占位，非真实 TTS"}


__all__ = ["SineEngine"]
