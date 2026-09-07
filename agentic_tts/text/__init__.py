"""
文本工程 / Text engineering — 送进模型之前该做的一切。

────────────────────────────────────────────────────────────────────────────
这一层存在的唯一理由
────────────────────────────────────────────────────────────────────────────
**开源 Qwen3-TTS 不支持 SSML，也不支持 LaTeX 朗读**（全仓库 grep 零命中）。
所以本地部署下，逐词的读法控制**只能在文本送进模型之前做完**。

三种手段里，这是唯一能做精细控制的一种：

    instruct 自然语言指令   → 整句的语气/语速（且 1.7B 才有）
    **文本预处理**          → 逐词的读法          ← 这一层
    标点与分段 + 插静音     → 停顿与节奏

────────────────────────────────────────────────────────────────────────────
设计约束
────────────────────────────────────────────────────────────────────────────
1. **全部是纯函数**：输入字符串，输出字符串，不碰模型、不碰 I/O。
   这是最终要移植进 DailyNewsAssistant 的部分，必须能脱离 TTS 单测到位。
2. **每条规则都该有实测依据**。过度归一化本身会引入问题
   （DNA issue 006-B：NFKC 把全角逗号折成半角，读感像机翻）。
   判定方法是「脏文本 vs 清洗后」的 CER 差，见 `cases/04_text_robustness.yaml`。
3. **顺序有依赖**，别自己拼流水线 —— 用 `prepare()`。

用法 / Usage::

    from agentic_tts.text import prepare, segment

    clean = prepare("## 今日要闻\\n\\n**通义千问**发布 Qwen3-TTS，WER 0.77。见 https://qwen.ai")
    for piece in segment(clean):
        wavs, sr = model.generate_custom_voice(text=piece, speaker="serena", ...)
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_tts.text import ascii_noise, lexicon, numbers, segment as _segment
from agentic_tts.text.ascii_noise import (
    collapse_repeated_punct,
    collapse_whitespace,
    strip_emoji,
    strip_markdown,
    strip_urls,
)
from agentic_tts.text.lexicon import (
    fix_polyphones,
    fix_proper_nouns,
    spell_abbreviations,
    spell_identifier,
)
from agentic_tts.text.numbers import normalise_all as normalise_numbers_all
from agentic_tts.text.segment import join_with_pauses, segment, split_sentences
from agentic_tts.text.events import EventPiece, known_events, parse_events, strip_events
from agentic_tts.text.split import TextPiece, split_text, unknown_roles


@dataclass(frozen=True)
class PrepareOptions:
    """
    每一步的开关 / Per-step switches.

    默认全开的只有「几乎总是对的」那几步。
    **多音字与专名默认关闭** —— 它们会把字换掉，且内置词表尚未逐条实测
    （见 `lexicon.py` 的说明），开着可能帮倒忙。
    Polyphone and proper-noun substitution are off by default: they alter the
    characters and the built-in tables are not yet measurement-backed.
    """

    markdown: bool = True
    urls: bool = True
    url_replacement: str = ""
    emoji: bool = True
    repeated_punct: bool = True
    numbers: bool = True
    abbreviations: bool = True
    polyphones: bool = False
    proper_nouns: bool = False
    whitespace: bool = True


def prepare(text: str, options: PrepareOptions | None = None) -> str:
    """
    按正确顺序跑完整条预处理流水线 / Run the full pipeline in the correct order.

    **顺序是有依赖的**，这是提供这个函数而不是让调用方自己拼的原因：

      1. Markdown 先剥 —— 否则 `[文字](https://…)` 里的 URL 会被 URL 规则
         先删掉，留下 `[文字]()` 这样的残骸；
      2. URL 再删 —— 必须在数字规则之前，否则 URL 里的数字会被念出来；
      3. emoji / 重复标点；
      4. 数字与单位（内部还有自己的六步顺序，见 `numbers.normalise_all`）；
      5. 缩写逐字母 —— 必须在数字之后，否则 `GB` 会先被拆成 `G B`
         而 `167GB` 的单位识别就失效了；
      6. 词典替换；
      7. 最后压空白。

    参数 / Args:
        text: 原文 / the source text.
        options: 每一步的开关 / per-step switches.

    返回 / Returns:
        可以直接送进 TTS 的文本 / text ready for the model.

    ⚠️ 如果开了 `polyphones` 或 `proper_nouns`，**返回的文本字已经变了**，
    只能送 TTS，不能拿去显示给人看。
    """
    options = options or PrepareOptions()

    if options.markdown:
        text = strip_markdown(text)
    if options.urls:
        text = strip_urls(text, options.url_replacement)
    if options.emoji:
        text = strip_emoji(text)
    if options.repeated_punct:
        text = collapse_repeated_punct(text)
    if options.numbers:
        text = normalise_numbers_all(text)
    if options.abbreviations:
        text = spell_abbreviations(text)
    if options.polyphones:
        text = fix_polyphones(text)
    if options.proper_nouns:
        text = fix_proper_nouns(text)
    if options.whitespace:
        text = collapse_whitespace(text)
    return text


def options_from_config(text_config) -> PrepareOptions:  # noqa: ANN001 - 避免循环 import
    """
    `TextConfig` → `PrepareOptions` / Map the config section onto the switches.

    只此一处做映射：两边字段名不完全一致（配置里叫 `lexicon_polyphones`，
    这里叫 `polyphones`），散着写迟早对不上。
    """
    return PrepareOptions(
        markdown=text_config.markdown,
        urls=text_config.urls,
        emoji=text_config.emoji,
        repeated_punct=text_config.repeated_punct,
        numbers=text_config.numbers,
        abbreviations=text_config.abbreviations,
        polyphones=text_config.lexicon_polyphones,
        proper_nouns=text_config.lexicon_proper_nouns,
    )


__all__ = [
    "EventPiece",
    "PrepareOptions",
    "TextPiece",
    "ascii_noise",
    "known_events",
    "options_from_config",
    "parse_events",
    "split_text",
    "strip_events",
    "unknown_roles",
    "collapse_repeated_punct",
    "collapse_whitespace",
    "fix_polyphones",
    "fix_proper_nouns",
    "join_with_pauses",
    "lexicon",
    "normalise_numbers_all",
    "numbers",
    "prepare",
    "segment",
    "spell_abbreviations",
    "spell_identifier",
    "split_sentences",
    "strip_emoji",
    "strip_markdown",
    "strip_urls",
]
