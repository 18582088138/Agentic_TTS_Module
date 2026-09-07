"""
Vocal events 标记 / Vocal-event markup —— `[pause:400ms]`、`[laugh]`、`[sigh]`…

────────────────────────────────────────────────────────────────────────────
先把事实说清楚：Qwen3-TTS 开源权重**没有**原生事件 token
────────────────────────────────────────────────────────────────────────────
核实方式两条（见 docs/00_research.md §3）：
  · `tokenizer_config.json` 的 special tokens 全表里只有 `<|im_start|>` /
    `<|audio_*|>` / `<|vision_*|>` 这类，没有 laugh / breath / sigh；
  · 上游仓库对 `laugh|breath|vocal|event` 全仓库 grep 零命中。

所以本层是**适配层**，把标记翻译成各引擎实际能做到的东西：

| 标记 | 做法 | 保证 |
|---|---|---|
| `[pause:400ms]` | 在该处切段，拼接时插等长静音 | ✅ 确定性，与引擎无关 |
| `[laugh]` 等 | 转成中文 instruct 提示追加给该段；文本里的标记删除 | ⚠️ 尽力而为 |
| 引擎声明 `VOCAL_EVENTS` | 标记原样留在文本里，交给引擎映射成原生标记 | 由引擎保证 |

**能力不足要说出来**：非原生时一律往 `warnings` 里写一条，
不能让调用方以为生效了 —— 静默降级正是上游 0.6B 丢弃 instruct 那个坑的形状。
Non-native handling always emits a warning instead of degrading silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# `[name]` 或 `[name:400ms]` / `[name:0.4s]` / `[name:400]`（不带单位按毫秒）
_MARKUP = re.compile(r"\[\s*([A-Za-z_一-鿿]+)\s*(?::\s*([0-9.]+)\s*(ms|s|秒|毫秒)?\s*)?\]")

# 事件 → instruct 提示（中文，1.7B 上 instruct 实测有效）
# Event → a Chinese instruct hint; instruct is measurably effective on the 1.7B.
_HINTS = {
    "laugh": "这句话带着笑意说，可以有轻笑声",
    "smile": "这句话带着笑意说",
    "sigh": "先叹一口气，语气低落",
    "breath": "开口前有一次明显的吸气",
    "cry": "带着哽咽，声音有些发抖",
    "whisper": "用耳语般的轻声说",
    "cough": "先轻咳一声",
    "gasp": "先有一次惊讶的吸气",
    "hmm": "先短促地嗯一声再说",
    "shout": "提高音量，像在喊",
    "yawn": "带着困倦的哈欠感",
}
# 中文别名，方便直接在稿子里写
_ALIASES = {
    "停顿": "pause", "笑": "laugh", "轻笑": "laugh", "叹气": "sigh", "吸气": "breath",
    "哽咽": "cry", "耳语": "whisper", "咳": "cough", "惊讶": "gasp", "喊": "shout",
    "哈欠": "yawn",
}

_PAUSE = "pause"


@dataclass
class EventPiece:
    """
    一段文本 + 它后面的停顿 + 它要追加的 instruct 提示 / One piece and its trailing pause.
    """

    text: str
    pause_after_ms: int = 0
    instruct_hints: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def _to_ms(value: str | None, unit: str | None, default_ms: int) -> int:
    if not value:
        return default_ms
    number = float(value)
    if unit in ("s", "秒"):
        return int(number * 1000)
    return int(number)


def parse_events(raw: str, *, default_pause_ms: int = 300,
                 native: bool = False) -> tuple[list[EventPiece], list[str]]:
    """
    解析事件标记 / Parse vocal-event markup.

    参数 / Args:
        raw: 带标记的文本 / text with markup.
        default_pause_ms: `[pause]` 不带时长时用多久 / default pause length.
        native: 引擎是否有**原生**事件支持。True 时非停顿标记原样保留，
            交给引擎；False 时删除标记并转成 instruct 提示。

    返回 / Returns:
        `(pieces, warnings)` —— `pieces` 按 `[pause]` 切开，顺序即朗读顺序；
        没有任何标记时就是一段。

    ⚠️ 停顿标记在**文本开头**时无处可挂（前面没有段），会被并进第一段之前的静音，
    也就是直接丢弃并记一条 warning —— 段首静音由拼接层的段间间隔负责，
    不该在这里再加一份。
    """
    warnings: list[str] = []
    pieces: list[EventPiece] = []
    current = EventPiece(text="")
    cursor = 0
    unknown: set[str] = set()

    for match in _MARKUP.finditer(raw):
        name = match.group(1).lower()
        name = _ALIASES.get(match.group(1), name)
        if name != _PAUSE and name not in _HINTS:
            # 不认识的方括号内容可能只是普通文本（如 [1]、[图]），原样留下
            unknown.add(match.group(0))
            continue

        current.text += raw[cursor:match.start()]
        cursor = match.end()

        if name == _PAUSE:
            ms = _to_ms(match.group(2), match.group(3), default_pause_ms)
            if not current.text.strip():
                warnings.append(f"段首的 {match.group(0)} 无处可挂，已忽略（段间停顿由拼接层负责）")
                current.text = ""
                continue
            current.pause_after_ms = ms
            current.events.append(f"pause:{ms}ms")
            pieces.append(current)
            current = EventPiece(text="")
            continue

        current.events.append(name)
        if native:
            current.text += match.group(0)      # 原生支持：标记留给引擎
        else:
            current.instruct_hints.append(_HINTS[name])

    current.text += raw[cursor:]
    if current.text.strip() or not pieces:
        pieces.append(current)

    if unknown:
        warnings.append("以下方括号内容不是已知事件标记，按普通文本处理："
                        + "、".join(sorted(unknown)))

    non_pause = [e for p in pieces for e in p.events if not e.startswith("pause")]
    if non_pause and not native:
        warnings.append(
            "本引擎无原生 vocal event 支持，"
            f"{'、'.join(sorted(set(non_pause)))} 已转成 instruct 提示，**效果不保证**")

    for piece in pieces:
        piece.text = piece.text.strip()
    return [p for p in pieces if p.text] or [EventPiece(text=raw.strip())], warnings


def strip_events(raw: str) -> str:
    """只去掉已知事件标记 / Remove known event markup, nothing else."""
    pieces, _ = parse_events(raw)
    return "".join(p.text for p in pieces)


def known_events() -> list[str]:
    """所有可用事件名（GUI 的下拉列表用）/ All event names, for the GUI picker."""
    return [_PAUSE, *sorted(_HINTS)]


__all__ = ["EventPiece", "known_events", "parse_events", "strip_events"]
