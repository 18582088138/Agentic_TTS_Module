"""
异常类型 / Exception types.

分四类，因为**调用方对这四类的处理方式不同**：配置错要改文件、能力错要换引擎、
加载错要看设备/显存、合成错可以重试。混成一个 Exception 就只能全都当致命错。
Four classes because callers react differently: fix the file, pick another engine,
check the device, or retry.
"""

from __future__ import annotations


class TTSError(Exception):
    """本模块所有异常的基类 / Base class for everything raised by this module."""


class ConfigError(TTSError):
    """配置不合法或路径不存在 / Invalid configuration or missing path."""


class CapabilityError(TTSError):
    """
    请求的能力当前引擎不支持 / The engine does not support the requested capability.

    **不静默降级**：上游 0.6B 静默丢弃 `instruct` 就是这个坑的形状 ——
    用例里写着指令却听不出变化，事后完全查不出原因。
    Never silently degrade; that is exactly how the upstream 0.6B `instruct` trap works.
    """


class EngineError(TTSError):
    """引擎加载失败（依赖缺失、权重缺失、显存不足）/ Engine failed to load."""


class SynthError(TTSError):
    """合成失败或结果不可用（含失控音频）/ Synthesis failed or produced unusable audio."""


__all__ = [
    "CapabilityError",
    "ConfigError",
    "EngineError",
    "SynthError",
    "TTSError",
]
