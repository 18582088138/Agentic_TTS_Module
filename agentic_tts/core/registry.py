"""
引擎注册表 / Engine registry —— 按名字懒加载。

`import agentic_tts` **永远不会拉起 torch**：只有真正选中的引擎才 import 它的重依赖，
而且各引擎的重依赖都在函数体里 import，所以
「查能力表」（GUI 开机要做的第一件事）也不需要 torch。
Importing the package never pulls torch; even the capability lookup stays light.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

from agentic_tts.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from agentic_tts.core.config import Config
    from agentic_tts.engines.base import TTSEngine

_ENGINES: dict[str, type] = {}

# 名字 → 模块。别名收在同一张表里，免得调用方猜。
_ENGINE_MODULES = {
    "qwen3": "agentic_tts.engines.qwen3",
    "qwen3_tts": "agentic_tts.engines.qwen3",
    "sine": "agentic_tts.engines.sine",
    "api": "agentic_tts.engines.api.engine",
    # 占位：接口未知，实例化时报清晰错误，但能力表已声明，GUI 可据此置灰
    "breeze2": "agentic_tts.engines.placeholders",
    "index2": "agentic_tts.engines.placeholders",
}


def register_engine(name: str) -> Callable[[type], type]:
    """类装饰器：注册一个引擎 / Register an engine class."""

    def deco(cls: type) -> type:
        _ENGINES[name] = cls
        return cls

    return deco


def get_engine_class(name: str) -> type["TTSEngine"]:
    """
    取引擎类（不实例化，因此不加载权重）/ Get the engine class without loading weights.

    抛出 / Raises:
        ConfigError: 名字不认识，或模块 import 失败。
    """
    key = (name or "").strip().lower()
    if key in _ENGINES:
        return _ENGINES[key]
    module = _ENGINE_MODULES.get(key)
    if module is None:
        raise ConfigError(
            f"不认识的引擎 {name!r}　可选：{'、'.join(sorted(set(_ENGINE_MODULES)))}")
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(f"引擎 {name!r} 的模块 {module} 导入失败：{exc}") from exc
    if key not in _ENGINES:
        raise ConfigError(f"模块 {module} 没有注册名为 {name!r} 的引擎（装饰器漏了？）")
    return _ENGINES[key]


def create_engine(name: str, config: "Config") -> "TTSEngine":
    """按名字造一个引擎实例 / Instantiate an engine. 权重仍是懒加载的。"""
    return get_engine_class(name)(config)


def engine_names() -> list[str]:
    """所有可选引擎名（含别名）/ All selectable engine names."""
    return sorted(set(_ENGINE_MODULES) | set(_ENGINES))


def engine_specs() -> list[dict]:
    """
    每个引擎的静态声明 / Static declaration per engine.

    **GUI 用这个置灰开关** —— 不加载任何权重就能知道谁支持克隆、谁支持指令。
    """
    specs = []
    for name in ("qwen3", "sine", "api", "breeze2", "index2"):
        try:
            cls = get_engine_class(name)
        except ConfigError:
            continue
        specs.append({
            "name": name,
            "implemented": bool(getattr(cls, "implemented", True)),
            "capabilities": sorted(c.value for c in cls.declared_capabilities()),
            "description": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        })
    return specs


__all__ = [
    "create_engine",
    "engine_names",
    "engine_specs",
    "get_engine_class",
    "register_engine",
]
