"""
把一整段文本拆成多段 / Split one blob of text into segments.

这是**给人用的拆分**（GUI 一键拆条、CLI `tts split`），与 `segment.py` 里
**给模型用的分段**是两件事，但它们共用同一套标点常量与同一个 `segment()` 实现 ——
两套分段规则会在长文案上给出不同切法，调试时看不出是哪套在起作用。
Authoring-level splitting shares one implementation with model-level segmentation.

四种规则 / Four rules:
    chars  按字数上限（仍在标点处落刀，不硬切）
    punct  一句一段
    role   按「角色：台词」前缀，顺带把角色映射到音色
    blank  按空行（作者已经分好段）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agentic_tts.core.types import SplitRule
from agentic_tts.text.segment import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, segment, split_sentences

# 「角色：台词」。全/半角冒号都认；角色名限长，避免把正文里的冒号误判成角色
# Role prefixes; the length cap keeps ordinary colons in prose from matching.
_ROLE_LINE = re.compile(r"^\s*(?:[【\[(（]?)([^：:【】\[\]()（）\n]{1,12})(?:[】\])）]?)\s*[：:]\s*(.+)$")


@dataclass
class TextPiece:
    """拆出来的一段 / One split-out piece."""

    text: str
    role: Optional[str] = None
    voice: Optional[str] = None


def split_text(
    text: str,
    rule: SplitRule | str = SplitRule.PUNCT,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = 0,
    role_voices: Optional[dict[str, str]] = None,
) -> list[TextPiece]:
    """
    按规则拆分 / Split by rule.

    参数 / Args:
        text: 一整段文本 / the whole blob.
        rule: `chars` | `punct` | `role` | `blank`.
        max_chars: 字数规则的上限 / upper bound for the `chars` rule.
        min_chars: 短于此长度的段落并进相邻段。**默认 0 = 不合并** ——
            这一层是"给人看的拆条"，说了一句一段就该是一句一段；
            合成时门面内部还会再按 `text.max_chars` 分块，碎片化在那一层解决。
            调大它（如 8）可以顺手把「好。」这种碎句并掉。
            Defaults to no merging: authoring-level splitting should do exactly
            what it says; model-level chunking happens later in the facade.
        role_voices: 角色名 → 音色名。**按角色拆分必须配这张表**，否则拆出来的
            对白没法配音（配置 `voices.roles` 与 GUI 都读同一张表）。

    返回 / Returns:
        段落列表，顺序即朗读顺序 / pieces in reading order.

    抛出 / Raises:
        ValueError: 规则名不认识。
    """
    rule = SplitRule(rule)
    roles = role_voices or {}

    if rule is SplitRule.ROLE:
        return _split_by_role(text, roles, max_chars=max_chars, min_chars=min_chars)
    if rule is SplitRule.BLANK:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        return [TextPiece(text=b.replace("\n", "")) for b in blocks]
    if rule is SplitRule.PUNCT:
        return [TextPiece(text=s) for s in _by_sentence(text, min_chars)]
    return [TextPiece(text=s) for s in segment(text, max_chars=max_chars, min_chars=min_chars)]


def _by_sentence(text: str, min_chars: int) -> list[str]:
    """一句一段，过短的并进上一段 / One sentence per piece, short ones merged back."""
    out: list[str] = []
    for block in (b for b in re.split(r"\n\s*\n", text) if b.strip()):
        for sentence in split_sentences(block.replace("\n", "")):
            sentence = sentence.strip()
            if not sentence:
                continue
            if out and len(sentence) < min_chars:
                out[-1] += sentence
            else:
                out.append(sentence)
    return out


def _split_by_role(text: str, role_voices: dict[str, str], *,
                   max_chars: int, min_chars: int) -> list[TextPiece]:
    """
    按「角色：台词」拆 / Split on `Role: line` prefixes.

    没有角色前缀的行归给**上一个**角色（对白里换行续写很常见）；
    第一行就没前缀时角色为 None，由调用方用默认音色兜。
    台词超长时按字数继续拆，但**拆出来的每一段仍带同一个角色**。
    """
    pieces: list[TextPiece] = []
    current_role: Optional[str] = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        voice = role_voices.get(current_role or "", None)
        for part in segment(body, max_chars=max_chars, min_chars=min_chars):
            pieces.append(TextPiece(text=part, role=current_role, voice=voice))

    for line in text.splitlines():
        if not line.strip():
            continue
        match = _ROLE_LINE.match(line)
        if match:
            flush()
            current_role = match.group(1).strip()
            buffer.append(match.group(2).strip())
        else:
            buffer.append(line.strip())
    flush()
    return pieces


def unknown_roles(pieces: list[TextPiece], role_voices: dict[str, str]) -> list[str]:
    """
    哪些角色没配音色 / Which roles have no voice mapped.

    GUI 用它提示「这几个角色会用默认音色」——**沉默地全用默认音色是最难查的 bug**：
    听起来所有人是同一个人，但没有任何报错。
    """
    return sorted({p.role for p in pieces if p.role and p.role not in role_voices})


__all__ = ["TextPiece", "split_text", "unknown_roles"]
