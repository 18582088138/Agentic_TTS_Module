"""
引擎实现 / Engine implementations.

**这里不 import 任何具体引擎** —— 由 `core.registry` 按名字懒加载，
于是 `import agentic_tts` 不会拉起 torch。
Engines are lazily imported by the registry so the package stays torch-free.
"""

from agentic_tts.engines.base import SynthChunk, TTSEngine

__all__ = ["SynthChunk", "TTSEngine"]
