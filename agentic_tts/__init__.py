"""
Agentic TTS Module —— 可插拔的本地 TTS 服务模块 / A pluggable local TTS service module.

默认引擎 Qwen3-TTS 1.7B（torch cpu/cuda），支持内置音色 / 音色设计 / 音色克隆
与 vocal events；长文本的规范化-分段-拼接-失控兜底由门面统一负责。

⚠️ 本文件**不 import torch，也不 import 任何引擎** ——
GUI 与 `--help` 都要能在不加载权重的前提下秒开。
This module imports neither torch nor any engine, so the CLI and GUI start instantly.
"""

from agentic_tts.core.config import Config
from agentic_tts.core.types import (
    BatchSynthRequest,
    Capability,
    Mode,
    SegmentResult,
    SplitRequest,
    SplitRule,
    SynthRequest,
    SynthResult,
    VoiceInfo,
)
from agentic_tts.engine import TTSModule

__version__ = "0.1.0"

__all__ = [
    "BatchSynthRequest",
    "Capability",
    "Config",
    "Mode",
    "SegmentResult",
    "SplitRequest",
    "SplitRule",
    "SynthRequest",
    "SynthResult",
    "TTSModule",
    "VoiceInfo",
    "__version__",
]
