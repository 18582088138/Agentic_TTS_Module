"""
API 后端引擎 / API backend engines.

抽象层：
- `base.py`        APIClient 协议 + BaseAPIClient（共享 voice cache + 错误处理）
- `minimax.py`     MiniMax adapter (speech-2.8-hd/turbo, voice_clone, voice_design)
- `engine.py`      APITTSEngine — 实现 TTSEngine 抽象，持有 APIClient

可独立测试：BaseAPIClient 的 voice cache、错误分类都是纯函数级别的。
HTTP 调用集中在各 provider adapter 里，函数体内 import httpx。
"""

from agentic_tts.engines.api.engine import APITTSEngine
from agentic_tts.engines.api.minimax import MiniMaxClient, register as _register

__all__ = ["APITTSEngine", "MiniMaxClient"]