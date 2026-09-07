"""
文本链路单元测试 / Text-pipeline unit tests —— 全纯函数，毫秒级。

复测 / Re-run:
    python -m pytest tests/test_text.py -q

覆盖什么 / What this covers:
  · 事件标记：`[pause]` 切段与时长换算、非原生时转 instruct 并给 warning
  · **标记必须先于文本预处理解析** —— 这是实测踩到的 bug（数字规范化把
    `[pause:500ms]` 写成了 `[pause:五百毫秒]`，停顿悄悄消失）
  · 四种拆分规则；角色规则的音色映射与「没配音色的角色」提示
  · 预处理里收益最大的两条（URL / Markdown）
"""

from __future__ import annotations

import pytest

from agentic_tts.core.types import SplitRule
from agentic_tts.text import prepare
from agentic_tts.text.events import known_events, parse_events, strip_events
from agentic_tts.text.split import split_text, unknown_roles


# ------------------------------------------------------------ 事件 / events


def test_pause_splits_into_pieces() -> None:
    pieces, warnings = parse_events("第一句话说完了。[pause:500ms]第二句才开始。")
    assert [p.text for p in pieces] == ["第一句话说完了。", "第二句才开始。"]
    assert pieces[0].pause_after_ms == 500
    assert pieces[1].pause_after_ms == 0
    assert not warnings


@pytest.mark.parametrize(
    ("markup", "expected_ms"),
    [("[pause]", 300), ("[pause:800ms]", 800), ("[pause:1.5s]", 1500), ("[pause:400]", 400)],
)
def test_pause_duration_units(markup, expected_ms) -> None:
    pieces, _ = parse_events(f"前半句。{markup}后半句。", default_pause_ms=300)
    assert pieces[0].pause_after_ms == expected_ms


def test_leading_pause_is_dropped_with_warning() -> None:
    """段首停顿无处可挂 —— 段间间隔由拼接层负责，不该再加一份。"""
    pieces, warnings = parse_events("[pause:500ms]开头就要停顿。")
    assert [p.text for p in pieces] == ["开头就要停顿。"]
    assert any("段首" in w for w in warnings)


def test_non_native_event_becomes_instruct_hint_and_warns() -> None:
    """
    不支持原生事件时必须**说出来**：静默降级正是上游 0.6B 丢弃 instruct 的坑形。
    """
    pieces, warnings = parse_events("这句话很有意思。[laugh]", native=False)
    assert "[laugh]" not in pieces[0].text
    assert pieces[0].instruct_hints and "笑" in pieces[0].instruct_hints[0]
    assert any("不保证" in w for w in warnings)


def test_native_engine_keeps_the_markup() -> None:
    pieces, warnings = parse_events("这句话很有意思。[laugh]", native=True)
    assert "[laugh]" in pieces[0].text
    assert not pieces[0].instruct_hints
    assert not any("不保证" in w for w in warnings)


def test_chinese_aliases_work() -> None:
    pieces, _ = parse_events("前半句。[停顿:600ms]后半句带笑。[笑]")
    assert pieces[0].pause_after_ms == 600
    assert pieces[1].instruct_hints


def test_ordinary_brackets_are_left_alone() -> None:
    """`[1]`、`[图2]` 明显不是事件标记（含数字），原样留下且不必啰嗦。"""
    pieces, warnings = parse_events("引用见 [1] 与 [图2]。")
    assert "[1]" in pieces[0].text and "[图2]" in pieces[0].text
    assert not warnings


def test_event_like_but_unknown_marker_warns() -> None:
    """
    `[laughing]` 长得就像事件标记却不是已知名字 —— 原样留下**并提示**，
    否则用户会以为它生效了，而实际上模型会把方括号连字母一起念出来。
    """
    pieces, warnings = parse_events("这句话很有意思 [laughing]。")
    assert "[laughing]" in pieces[0].text
    assert any("方括号" in w for w in warnings)


def test_strip_events() -> None:
    assert strip_events("你好[laugh]世界") == "你好世界"


def test_known_events_includes_pause_first() -> None:
    assert known_events()[0] == "pause"


# ------------------------------------------ 标记与预处理的顺序 / ordering bug


def test_number_normaliser_would_eat_the_markup() -> None:
    """
    实测踩到的那个 bug 的**回归用例**：先跑预处理会把 `500ms` 换成中文，
    于是 `[pause:500ms]` 再也匹配不上。门面里的顺序必须是「先解析标记」。
    Regression: preparing before parsing rewrites `500ms` inside the markup.
    """
    mangled = prepare("前半句。[pause:500ms]后半句。")
    assert "[pause:500ms]" not in mangled          # 预处理确实会改掉它
    pieces, _ = parse_events(mangled)
    assert len(pieces) == 1                        # 于是停顿丢了 —— 顺序反了就是这个后果

    pieces, _ = parse_events("前半句。[pause:500ms]后半句。")
    assert len(pieces) == 2                        # 正确顺序：先解析


# ------------------------------------------------------------ 预处理 / prepare


def test_url_is_removed() -> None:
    """URL 是收益最大的一条：不处理时 CER 4.43（模型逐字母念网址）。"""
    assert "qwen.ai" not in prepare("详情见 https://qwen.ai/blog 。")


def test_markdown_is_stripped() -> None:
    out = prepare("## 今日要闻\n\n**通义千问**发布了新模型")
    assert "#" not in out and "**" not in out
    assert "通义千问" in out


# ------------------------------------------------------------- 拆分 / split


def test_split_by_punct() -> None:
    pieces = split_text("第一句话。第二句话！第三句话？", SplitRule.PUNCT)
    assert [p.text for p in pieces] == ["第一句话。", "第二句话！", "第三句话？"]


def test_split_by_chars_respects_upper_bound() -> None:
    text = "".join(f"这是第{i}句话，内容比较长一些。" for i in range(1, 9))
    pieces = split_text(text, SplitRule.CHARS, max_chars=30, min_chars=8)
    assert len(pieces) > 1
    assert all(len(p.text) <= 45 for p in pieces)   # 标点处落刀，允许略超上限


def test_split_by_blank_lines() -> None:
    pieces = split_text("第一段的内容。\n\n第二段的内容。", SplitRule.BLANK)
    assert len(pieces) == 2


def test_split_by_role_maps_voices() -> None:
    text = "旁白：今天的资讯。\n记者：第一条消息。\n继续说下去。"
    pieces = split_text(text, SplitRule.ROLE, role_voices={"旁白": "Serena", "记者": "Ryan"})
    assert [p.role for p in pieces] == ["旁白", "记者"]
    assert [p.voice for p in pieces] == ["Serena", "Ryan"]
    # 没有前缀的行归给上一个角色（对白续写很常见）
    assert "继续说下去" in pieces[1].text


def test_split_by_role_reports_unmapped_roles() -> None:
    """沉默地全用默认音色 = 听起来所有人是同一个人，且没有任何报错。"""
    pieces = split_text("甲：第一句。\n乙：第二句。", SplitRule.ROLE, role_voices={"甲": "Serena"})
    assert unknown_roles(pieces, {"甲": "Serena"}) == ["乙"]


def test_role_prefix_does_not_eat_ordinary_colons() -> None:
    """正文里的冒号（尤其长句）不该被误判成角色前缀。"""
    text = "他说了一句很长的话里面带了一个冒号：这只是普通的正文内容而不是角色名。"
    pieces = split_text(text, SplitRule.ROLE, role_voices={})
    assert all(p.role is None for p in pieces)
