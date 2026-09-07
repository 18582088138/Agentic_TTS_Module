"""
ASCII 噪声清理 / ASCII noise removal.

处理「不该被念出来」的东西：URL、Markdown 标记、emoji、颜文字、重复标点。

────────────────────────────────────────────────────────────────────────────
为什么这一层优先级最高
────────────────────────────────────────────────────────────────────────────
数字读错只是一个词不对，**URL 被念出来是整段毁掉**：
`https://qwen.ai/blog?id=qwen3tts-0115` 会变成十几秒的字母流。

而 Markdown 残留是必然会有的 —— DailyNewsAssistant 的脚本由 LLM 生成，
`**加粗**`、`## 标题`、`[文字](链接)` 一定会漏出来。

全部是纯函数：输入字符串，输出字符串，不碰模型、不碰 I/O。
Pure string-in string-out; no model, no I/O.
"""

from __future__ import annotations

import re

# ── URL ──────────────────────────────────────────────────────────────────
# 带协议的、以及 www. 开头的裸域名。
# 尾部的标点不吃进来（`见 https://a.com。` 里的句号要留着断句）。
_URL = re.compile(
    r"""(?ix)
    \b
    (?: https? :// | www\. )
    [^\s一-鿿，。！？；：、（）《》""'']+
    """
)

# ── Markdown ─────────────────────────────────────────────────────────────
# 顺序有讲究：先处理链接（要保留文字、丢掉 URL），再剥行内标记。
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_QUOTE = re.compile(r"(?m)^\s{0,3}>\s?")
_MD_LIST = re.compile(r"(?m)^\s{0,3}(?:[-*+]|\d+\.)\s+")
_MD_RULE = re.compile(r"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
# 粗体/斜体/删除线。`**x**` 与 `*x*` 都要，且不能吃掉数学里的乘号。
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_MD_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_MD_STRIKE = re.compile(r"~~([^~]+)~~")

# ── emoji 与颜文字 ───────────────────────────────────────────────────────
# 覆盖主要的 emoji 区段 + 变体选择符 + 零宽连接符。
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # 符号与图形、补充符号
    "\U0001fa00-\U0001faff"  # 扩展 A
    "\U00002600-\U000027bf"  # 杂项符号与装饰符
    "\U0001f1e6-\U0001f1ff"  # 区域指示符（国旗）
    "←-⇿"          # 箭头
    "⬀-⯿"
    "︀-️"          # 变体选择符
    "‍"                 # 零宽连接符
    "]+"
)
# 颜文字：括号里塞满非中英文的符号。`(◍•͈⌔•͈◍)` 是 README 里的实例。
_KAOMOJI = re.compile(r"[（(][^\w\s一-鿿]{2,}[)）]")

# ── 重复标点 ─────────────────────────────────────────────────────────────
_REPEATED_PUNCT = re.compile(r"([！？!?。.，,、；;：:…])\1{1,}")

# ── 空白 ─────────────────────────────────────────────────────────────────
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 中文标点前的空格。删掉 URL 或 emoji 之后必然留下这种残迹：
# `详见 https://…。` → `详见 。`。空格本身不发声，但它会让分段与
# 标点识别错位，也让 CER 的对齐多出噪声。
# Left behind whenever a URL or emoji is removed before punctuation.
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[，。！？；：、）》」』])")
_SPACE_AFTER_OPEN = re.compile(r"(?<=[（《「『])[ \t]+")


def strip_urls(text: str, replacement: str = "") -> str:
    """
    删掉 URL / Remove URLs.

    **URL 必须删。** 念出来是十几秒的字母流，整段就毁了。

    参数 / Args:
        text: 原文 / the source text.
        replacement: 替换成什么。默认删干净；想让听众知道有个链接，
            可以传 `"链接见简介"` / what to put in place; empty removes them.

    返回 / Returns:
        处理后的文本 / the processed text.

    示例 / Example::

        >>> strip_urls("详情见 https://qwen.ai/blog?id=x，模型很强。")
        '详情见 ，模型很强。'
    """
    return _URL.sub(replacement, text)


def strip_markdown(text: str) -> str:
    """
    剥掉 Markdown 标记，保留文字 / Strip Markdown markup, keep the words.

    DailyNewsAssistant 的脚本由 LLM 生成，`**加粗**` `## 标题` `[文字](链接)`
    一定会漏出来 —— 这一步是必做的。
    LLM-generated scripts always leak Markdown, so this step is mandatory.

    链接**保留文字、丢掉 URL**（`[官方博客](https://…)` → `官方博客`），
    因为链接文字往往是句子的一部分，删了句子就断了。
    Link text is kept because it is usually part of the sentence.

    示例 / Example::

        >>> strip_markdown("## 今日要闻\\n\\n**通义千问**发布了 `Qwen3-TTS`。")
        '今日要闻\\n\\n通义千问发布了 Qwen3-TTS。'
    """
    text = _MD_CODE_BLOCK.sub(" ", text)
    text = _MD_IMAGE.sub(r"\1", text)   # 图片：alt 文字通常没有朗读价值，但也不至于有害
    text = _MD_LINK.sub(r"\1", text)    # 链接：保留文字，丢掉 URL
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_STRIKE.sub(r"\1", text)
    return text


def strip_emoji(text: str) -> str:
    """
    删掉 emoji 与颜文字 / Remove emoji and kaomoji.

    README 用 `(◍•͈⌔•͈◍)` 展示模型「遇到脏文本不会崩」，
    但**不崩不等于该留着** —— 它要么被跳过，要么被念成奇怪的东西。
    Surviving noisy input is not a reason to keep the noise.
    """
    return _KAOMOJI.sub("", _EMOJI.sub("", text))


def collapse_repeated_punct(text: str, keep: int = 1) -> str:
    """
    把重复的标点压缩 / Collapse repeated punctuation.

    `真的吗？？？` → `真的吗？`。重复标点在文本里表示强调，
    但模型看到的是三个独立的问号，可能读出三次停顿。
    Repeated marks signal emphasis to a reader but become three pauses to the model.

    参数 / Args:
        keep: 保留几个 / how many to keep.
    """
    if keep < 1:
        raise ValueError("keep 至少为 1")
    return _REPEATED_PUNCT.sub(lambda m: m.group(1) * keep, text)


def collapse_whitespace(text: str) -> str:
    """
    压缩多余空白，但**保留换行结构** / Collapse whitespace, keeping line structure.

    换行是分段的依据（`segment.py` 会用到），不能一并压掉。
    Newlines carry the paragraph structure that the segmenter relies on.

    顺带清掉**中文标点前后的游离空格** —— 删 URL 或 emoji 之后必然留下这种残迹
    （`详见 https://…。` → `详见 。`）。空格不发声，但会让分段与标点识别错位。
    """
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    text = _SPACE_AFTER_OPEN.sub("", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


__all__ = [
    "collapse_repeated_punct",
    "collapse_whitespace",
    "strip_emoji",
    "strip_markdown",
    "strip_urls",
]
