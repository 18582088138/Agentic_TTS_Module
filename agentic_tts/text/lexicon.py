"""
词典替换：多音字、专名、英文缩写 / Dictionary substitution.

────────────────────────────────────────────────────────────────────────────
为什么只能用词典
────────────────────────────────────────────────────────────────────────────
**开源 Qwen3-TTS 没有 SSML，也没有拼音标注接口。**
所以「这个字该读哪个音」这件事，没有任何办法告诉模型 ——
唯一的手段是**把字换掉**，换成一个读音无歧义的同音字。

    重庆 → 崇庆     （重 chóng，不是 zhòng）
    银行 → 银航     （行 háng，不是 xíng）

代价是**字变了**，所以：

  1. 只对**读错概率高**的词做，不做全量替换；
  2. 替换后的文本**只用于送进 TTS**，不能拿去显示给人看；
  3. 每加一条都该有实测依据（跑 `04_text_robustness.yaml` 看 CER 有没有改善），
     而不是凭感觉往里加 —— 替换本身也可能引入新问题。

────────────────────────────────────────────────────────────────────────────
⚠️ 这里的词条是**起点，不是结论**
────────────────────────────────────────────────────────────────────────────
下面的词条来自「常见多音字」的一般知识，**尚未在 4060 上逐条实测**。
实测之后该删的删（模型本来就读对的不用替换）、该加的加。
在验证之前，把它们当作待验证的假设，不要当作已知结论。
The entries below are hypotheses awaiting measurement, not verified conclusions.
"""

from __future__ import annotations

import re

# ── 多音字 ───────────────────────────────────────────────────────────────
# 键是原词，值是同音替换。**按词而不是按字**替换 —— 单字没有上下文，
# 「重」到底读哪个音只有在词里才定得下来。
# Keyed by word, not by character: a lone 重 has no disambiguating context.
POLYPHONE: dict[str, str] = {
    # 重 chóng vs zhòng
    "重庆": "崇庆",
    "重复": "崇复",
    "重新": "崇新",
    "重叠": "崇叠",
    # 行 háng vs xíng
    "银行": "银航",
    "行业": "航业",
    "同行": "同航",
    "排行": "排航",
    # 长 cháng vs zhǎng
    "长度": "常度",
    "长文本": "常文本",
    # 模 mó vs mú
    "模型": "摩型",
    "模式": "摩式",
    # 差 chā vs chà vs chāi
    "误差": "误插",
    "差异": "插异",
    # 数 shù vs shǔ
    "参数": "参树",
    "数据": "树据",
    # 处 chù vs chǔ
    "处理": "础理",
    # 分 fēn vs fèn
    "分辨": "芬辨",
    # 相 xiāng vs xiàng
    "相似": "香似",
    "相对": "香对",
}

# ── 专名 ─────────────────────────────────────────────────────────────────
# 中文专名：模型容易读错或断错的。
# 英文专名：**给出中文读法**，避免模型按英文发音规则乱读。
PROPER_NOUNS: dict[str, str] = {
    "通义千问": "通义千问",  # 占位：实测确认读法之后再改
    "OpenVINO": "欧朋维诺",
    "HuggingFace": "抱脸",
    "ModelScope": "魔搭",
    "PyTorch": "派托奇",
    "NVIDIA": "英伟达",
    "GitHub": "吉特哈布",
}

# ── 英文缩写 → 逐字母读 ──────────────────────────────────────────────────
# 这些是**当字母念**的，不是当单词念。模型有可能把 `API` 念成 "appy"。
# 值用空格分开，让模型知道是逐个字母。
ABBREVIATIONS: dict[str, str] = {
    "TTS": "T T S",
    "ASR": "A S R",
    "API": "A P I",
    "WER": "W E R",
    "CER": "C E R",
    "RTF": "R T F",
    "GPU": "G P U",
    "CPU": "C P U",
    "NPU": "N P U",
    "LLM": "L L M",
    "SSML": "S S M L",
    "URL": "U R L",
    "JSON": "J S O N",
    "YAML": "Y A M L",
    "GB": "G B",
    "MB": "M B",
    "KB": "K B",
    "TB": "T B",
    "RTX": "R T X",
    "GTX": "G T X",
    "AI": "A I",
    "ID": "I D",
    "OK": "O K",
}


def _compile(mapping: dict[str, str], *, word_boundary: bool) -> re.Pattern[str] | None:
    """
    把词典编译成一个正则 / Compile a dictionary into one alternation.

    **按长度倒序**，否则短词会先匹配掉长词的一部分
    （`重复` 与 `重` 同时在表里时，必须先试 `重复`）。
    Sorted longest-first so a short key cannot pre-empt a longer one.
    """
    if not mapping:
        return None
    keys = sorted(mapping, key=len, reverse=True)
    body = "|".join(re.escape(k) for k in keys)
    # 中文没有词边界的概念，`\b` 对中文无效；英文缩写才需要边界，
    # 否则 `AI` 会命中 `TRAIN` 里的 `AI`。
    return re.compile(rf"\b(?:{body})\b" if word_boundary else f"(?:{body})")


_POLYPHONE_RE = _compile(POLYPHONE, word_boundary=False)
_PROPER_RE = _compile(PROPER_NOUNS, word_boundary=False)
_ABBREV_RE = _compile(ABBREVIATIONS, word_boundary=True)


def apply_lexicon(
    text: str, mapping: dict[str, str], *, word_boundary: bool = False
) -> str:
    """
    按词典替换 / Substitute according to a dictionary.

    参数 / Args:
        text: 原文 / the source text.
        mapping: 词典 / the dictionary.
        word_boundary: 英文词条设 True（否则 `AI` 会命中 `TRAIN`）；
            中文词条设 False（`\\b` 对中文无效）。

    返回 / Returns:
        替换后的文本 / the substituted text.
    """
    pattern = _compile(mapping, word_boundary=word_boundary)
    if pattern is None:
        return text
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def fix_polyphones(text: str, extra: dict[str, str] | None = None) -> str:
    """
    多音字替换 / Substitute polyphonic words.

    ⚠️ **替换后的文本只能送 TTS，不能拿去显示** —— 字已经变了。

    参数 / Args:
        extra: 额外词条，会覆盖内置表 / extra entries, overriding the built-ins.
    """
    mapping = {**POLYPHONE, **(extra or {})}
    return apply_lexicon(text, mapping, word_boundary=False)


def fix_proper_nouns(text: str, extra: dict[str, str] | None = None) -> str:
    """专名读法 / Proper-noun readings."""
    mapping = {**PROPER_NOUNS, **(extra or {})}
    return apply_lexicon(text, mapping, word_boundary=False)


def spell_abbreviations(text: str, extra: dict[str, str] | None = None) -> str:
    """
    英文缩写改成逐字母 / Spell out English abbreviations.

    `TTS` → `T T S`。用空格分开，让模型知道是逐个字母而不是一个单词。
    """
    mapping = {**ABBREVIATIONS, **(extra or {})}
    return apply_lexicon(text, mapping, word_boundary=True)


def spell_identifier(identifier: str) -> str:
    """
    代码标识符逐字符拆开 / Spell out a code identifier.

    `UD-Q8_K_XL` → `U D Q 8 K X L`。AI 资讯里这类东西高频出现，
    不拆开的话模型会试图把它当一个词读。
    Frequent in AI news; unsplit, the model tries to read it as a word.

    示例 / Example::

        >>> spell_identifier("UD-Q8_K_XL")
        'U D Q 8 K X L'
    """
    return " ".join(c.upper() for c in identifier if c.isalnum())


__all__ = [
    "ABBREVIATIONS",
    "POLYPHONE",
    "PROPER_NOUNS",
    "apply_lexicon",
    "fix_polyphones",
    "fix_proper_nouns",
    "spell_abbreviations",
    "spell_identifier",
]
