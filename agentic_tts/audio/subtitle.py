"""
字幕导出 / Subtitle export —— 一段音频对应一条字幕，输出 SRT。

**为什么是 SRT**：剪映、Premiere、DaVinci、PotPlayer 都能直接导入 SRT，
而它本身就是纯文本，没有依赖、没有版本问题。所以不需要退化成 txt。
SRT is imported directly by JianYing/Premiere/Resolve and is plain text.

时间轴按**拼接后的整条音频**算：第 i 条的开始时间 = 前面所有段的时长与段间静音之和。
段间静音必须算进去 —— 少算的话越往后字幕越提前，尾部能差出好几秒。
The timeline includes the inter-segment silence; omitting it makes later cues drift early.

字幕文字用**原文**（去掉事件标记），不是送进模型的那份：
预处理会把 URL 删掉、把数字念法展开（`3.5` → `三点五`），那是给模型看的，
显示给人看会很怪。
Cues carry the original text, not the model-facing text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    """一条字幕 / One subtitle cue（秒）。"""

    start: float
    end: float
    text: str


def build_cues(durations: list[float], texts: list[str],
               gaps_ms: list[int] | None = None) -> list[Cue]:
    """
    按时长与段间静音排出时间轴 / Lay out the timeline.

    参数 / Args:
        durations: 每段音频时长（秒）。
        texts: 每段的字幕文字（与 durations 一一对应）。
        gaps_ms: 每段**之后**的静音毫秒；最后一项被忽略（结尾不拖静音）。
            None = 段间无间隔。

    返回 / Returns:
        `list[Cue]`，空文本的段会被跳过（不产生空字幕）。

    抛出 / Raises:
        ValueError: 三个列表长度对不上。
    """
    if len(durations) != len(texts):
        raise ValueError(f"长度对不上：durations={len(durations)} texts={len(texts)}")
    gaps = list(gaps_ms) if gaps_ms is not None else [0] * len(durations)
    if len(gaps) != len(durations):
        raise ValueError(f"长度对不上：gaps={len(gaps)} durations={len(durations)}")

    cues: list[Cue] = []
    offset = 0.0
    for index, duration in enumerate(durations):
        text = (texts[index] or "").strip()
        if text:
            cues.append(Cue(start=offset, end=offset + duration, text=text))
        offset += duration
        if index < len(durations) - 1:
            offset += gaps[index] / 1000.0
    return cues


def _stamp(seconds: float) -> str:
    """秒 → `HH:MM:SS,mmm`（SRT 用逗号作小数点，不是句点）。"""
    if seconds < 0:
        seconds = 0.0
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def to_srt(cues: list[Cue]) -> str:
    """
    渲染成 SRT 文本 / Render SRT text.

    序号从 1 开始、块之间空一行、**结尾留一个换行** —— 有些编辑器对缺少末尾换行的
    SRT 会丢掉最后一条。
    Some editors drop the final cue if the file does not end with a newline.
    """
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(f"{index}\n{_stamp(cue.start)} --> {_stamp(cue.end)}\n{cue.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: str | Path, cues: list[Cue]) -> Path:
    """
    落盘 / Write the SRT file.

    **UTF-8 带 BOM**：剪映与部分 Windows 工具读不带 BOM 的中文 SRT 会乱码。
    BOM 对能正确识别 UTF-8 的工具无影响。
    UTF-8 **with BOM**: JianYing and some Windows tools mis-decode Chinese without it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_srt(cues), encoding="utf-8-sig")
    return target


__all__ = ["Cue", "build_cues", "to_srt", "write_srt"]
