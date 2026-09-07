"""
长文本分段 / Long-text segmentation.

────────────────────────────────────────────────────────────────────────────
为什么必须分段 —— 有数据支撑，不是习惯
────────────────────────────────────────────────────────────────────────────
1. **准确率**：12Hz 模型在长文本上的字错误率是 2.356，短句上是 0.903。
   长文本上明显吃亏，分段直接把这一项拉回来。
2. **节奏**：没有 SSML，停顿只能靠标点和**段间插静音**。
   静音时长是确定的，不依赖模型发挥 —— 这是最可靠的节奏控制手段。
3. **容错**：批量生成是要么全成要么全败。几十段的长文案跑到第 40 段崩掉，
   会赔上前面 39 段的时间。逐段来还能报进度。

分段是纯函数；拼接与插静音需要 numpy，放在 `join_with_pauses`。
"""

from __future__ import annotations

import re

# 中文句末标点。`；` 也算 —— 分号在中文里的停顿接近句号。
_SENTENCE_END = "。！？；!?;"
# 分句：在句末标点之后切，标点跟着前一句走
_SPLIT_SENTENCE = re.compile(rf"(?<=[{re.escape(_SENTENCE_END)}])")
# 次级切点：逗号、顿号。只在句子超长时才用。
_SECONDARY = re.compile(r"(?<=[，,、])")

# 一段的目标长度。依据：Qwen3-TTS 的 Seed-TTS 测试集句子多在这个量级，
# 而 benchmark 里「长文本」那一栏的劣化就是从更长的段落开始的。
DEFAULT_MAX_CHARS = 60
# 低于这个长度的段落会被并进相邻段 —— 一个三字的段落单独合成，
# 前后各插一段静音，听起来是突兀的停顿。
DEFAULT_MIN_CHARS = 8

# 段间静音（秒）。同一说话人用短的，换人用长的 —— 换人需要一个可感知的间隔，
# 否则听众分不清是同一个人换了语气还是换了人。
# 与 DailyNewsAssistant 的 PAUSE_SECONDS / SPEAKER_CHANGE_PAUSE_SECONDS 对齐。
PAUSE_SECONDS = 0.3
SPEAKER_CHANGE_PAUSE_SECONDS = 0.6


def split_sentences(text: str) -> list[str]:
    """
    按句末标点切句 / Split on sentence-ending punctuation.

    标点**跟着前一句走** —— 模型需要它来决定句尾的语调。
    Punctuation stays with the preceding sentence; the model needs it for intonation.

    示例 / Example::

        >>> split_sentences("第一句。第二句！第三句")
        ['第一句。', '第二句！', '第三句']
    """
    return [s.strip() for s in _SPLIT_SENTENCE.split(text) if s.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """一句话本身就超长时，退而在逗号处切 / Fall back to commas for an over-long sentence."""
    if len(sentence) <= max_chars:
        return [sentence]

    chunks = [c for c in _SECONDARY.split(sentence) if c.strip()]
    pieces: list[str] = []
    buffer = ""
    for chunk in chunks:
        if buffer and len(buffer) + len(chunk) > max_chars:
            pieces.append(buffer)
            buffer = chunk
        else:
            buffer += chunk
    if buffer:
        pieces.append(buffer)

    # 连逗号都没有的超长句：**不硬切**。
    # 在没有标点的地方切开会把词切断，模型读出来是断裂的 ——
    # 宁可让这一段长一点，也不要制造一个读不通的片段。
    # A hard cut mid-word produces audibly broken speech; an over-long segment is better.
    return pieces or [sentence]


def segment(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """
    把长文本切成适合逐段合成的片段 / Segment long text for piece-by-piece synthesis.

    策略 / Strategy:
      1. 先按句末标点切句；
      2. 短句往前合并，直到接近 `max_chars`（避免碎片化）；
      3. 单句超过 `max_chars` 时在逗号处再切；
      4. 结尾如果剩一个过短的片段，并回上一段。

    参数 / Args:
        text: 原文 / the source text.
        max_chars: 一段的目标上限 / target upper bound per segment.
        min_chars: 低于此长度的段落会被并进相邻段 / segments shorter than this
            are merged, because a three-character segment surrounded by silence
            sounds like a stumble.

    返回 / Returns:
        片段列表，顺序即朗读顺序 / segments in reading order.

    抛出 / Raises:
        ValueError: `max_chars` 不大于 `min_chars` —— 那样会陷入死循环般的碎片化。
    """
    if max_chars <= min_chars:
        raise ValueError(f"max_chars({max_chars}) 必须大于 min_chars({min_chars})")

    # 空行分段是作者的意图，先按它切，再在每块里分句
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    segments: list[str] = []

    for block in blocks:
        buffer = ""
        for sentence in split_sentences(block.replace("\n", "")):
            for piece in _split_long(sentence, max_chars):
                if not buffer:
                    buffer = piece
                elif len(buffer) + len(piece) <= max_chars:
                    buffer += piece
                else:
                    segments.append(buffer)
                    buffer = piece
        if buffer:
            segments.append(buffer)

    # 收尾：把过短的片段并进相邻段。
    # **向前并不够** —— 开头的短片段（多半是标题，如「今日要闻」）前面没有东西可并，
    # 会孤零零地留下来，合成后前后各插一段静音，听起来是个突兀的停顿。
    # 所以它要向**后**并进下一段。已实测踩到。
    # Merging backwards alone strands a short leading segment such as a heading.
    merged: list[str] = []
    for piece in segments:
        if merged and len(piece) < min_chars:
            merged[-1] += piece
        else:
            merged.append(piece)

    if len(merged) > 1 and len(merged[0]) < min_chars:
        head = merged.pop(0)
        # 标题与正文之间补一个句号，否则两句会连读成一句
        separator = "" if head[-1] in _SENTENCE_END + "，,、" else "。"
        merged[0] = head + separator + merged[0]

    return merged


def join_with_pauses(
    clips: list,
    sample_rate: int,
    *,
    roles: list[str] | None = None,
    pause: float = PAUSE_SECONDS,
    speaker_change_pause: float = SPEAKER_CHANGE_PAUSE_SECONDS,
):
    """
    拼接多段音频，段间插静音 / Concatenate clips with silence between them.

    **这是没有 SSML 时最可靠的节奏控制手段** —— 静音时长是确定的，
    不依赖模型发挥。
    The one rhythm control that does not depend on the model's behaviour.

    参数 / Args:
        clips: 一维 float32 波形列表 / list of 1-D float32 waveforms.
        sample_rate: 采样率 / sample rate.
        roles: 每段的说话人。给了就在换人处用更长的停顿 / speaker per clip.
        pause: 同一说话人的段间静音（秒）/ silence between same-speaker segments.
        speaker_change_pause: 换人时的静音（秒）/ silence at a speaker change.

    返回 / Returns:
        拼接后的一维波形 / the concatenated waveform.

    抛出 / Raises:
        ValueError: `clips` 为空，或 `roles` 长度对不上。
    """
    import numpy as np

    if not clips:
        raise ValueError("没有可拼接的音频")
    if roles is not None and len(roles) != len(clips):
        raise ValueError(f"roles 长度 {len(roles)} 与 clips 长度 {len(clips)} 对不上")

    pieces: list = []
    previous_role: str | None = None
    for index, clip in enumerate(clips):
        if index:
            role = roles[index] if roles else None
            gap = (
                speaker_change_pause
                if roles and role != previous_role
                else pause
            )
            pieces.append(np.zeros(int(sample_rate * gap), dtype=np.float32))
        previous_role = roles[index] if roles else None
        pieces.append(np.asarray(clip, dtype=np.float32).reshape(-1))

    return np.concatenate(pieces)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_CHARS",
    "PAUSE_SECONDS",
    "SPEAKER_CHANGE_PAUSE_SECONDS",
    "join_with_pauses",
    "segment",
    "split_sentences",
]
