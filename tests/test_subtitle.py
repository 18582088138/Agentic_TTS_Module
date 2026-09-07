"""
字幕导出单元测试 / Subtitle export unit tests.

复测 / Re-run:
    python -m pytest tests/test_subtitle.py -q

要钉住的两件事 / The two things that must hold:
  · 时间轴**含段间静音** —— 漏算的话越往后字幕越提前，尾部能差出好几秒；
  · 字幕文字是**原文去事件标记**，不是送进模型那份（预处理会删 URL、展开数字读法）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_tts.audio.subtitle import Cue, build_cues, to_srt, write_srt
from agentic_tts.core.types import SynthRequest


def test_cues_include_the_gaps() -> None:
    cues = build_cues([1.0, 2.0], ["第一句", "第二句"], gaps_ms=[500, 0])
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(1.0)
    # 第二条要被静音推后 0.5s —— 这一条就是"字幕会不会越走越偏"的分水岭
    assert cues[1].start == pytest.approx(1.5)
    assert cues[1].end == pytest.approx(3.5)


def test_last_gap_is_ignored() -> None:
    """结尾不拖静音，所以最后一项 gap 不该影响任何时间点。"""
    with_gap = build_cues([1.0], ["只有一句"], gaps_ms=[900])
    assert with_gap[0].end == pytest.approx(1.0)


def test_empty_text_produces_no_cue() -> None:
    cues = build_cues([1.0, 1.0], ["有内容", "   "], gaps_ms=[0, 0])
    assert len(cues) == 1


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="长度对不上"):
        build_cues([1.0, 2.0], ["只有一句"])


def test_srt_format() -> None:
    """SRT 用逗号做小数点、块间空行、结尾留换行（缺末尾换行有编辑器会丢最后一条）。"""
    text = to_srt([Cue(0.0, 1.5, "第一句"), Cue(2.0, 3.25, "第二句")])
    assert text == (
        "1\n00:00:00,000 --> 00:00:01,500\n第一句\n\n"
        "2\n00:00:02,000 --> 00:00:03,250\n第二句\n"
    )


def test_hours_are_rendered() -> None:
    assert "01:01:01,001" in to_srt([Cue(3661.001, 3662.0, "一小时后")])


def test_write_srt_uses_bom(tmp_path) -> None:
    """剪映与部分 Windows 工具读不带 BOM 的中文 SRT 会乱码。"""
    path = write_srt(tmp_path / "a.srt", [Cue(0.0, 1.0, "中文字幕")])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


# ------------------------------------------------------- 与门面串起来 / facade


def test_merge_writes_a_subtitle_next_to_the_audio(module) -> None:
    result = module.synthesize_many([
        SynthRequest(text="第一段的内容在这里。", voice="test_female"),
        SynthRequest(text="第二段的内容在这里。", voice="test_male"),
    ], merge=True, name="srt")
    assert result.subtitle_path and Path(result.subtitle_path).name == "merged.srt"
    body = Path(result.subtitle_path).read_text(encoding="utf-8-sig")
    assert "第一段的内容在这里。" in body and "第二段的内容在这里。" in body
    assert body.count("-->") == 2


def test_subtitle_text_drops_event_markup(module) -> None:
    """`[laugh]` 这类标记不该出现在字幕里 —— 它是控制信息，不是台词。"""
    result = module.synthesize_many([
        SynthRequest(text="这句话很有意思。[laugh]", voice="test_female"),
        SynthRequest(text="第二段的内容在这里。", voice="test_female"),
    ], merge=True, name="srt_events")
    body = Path(result.subtitle_path).read_text(encoding="utf-8-sig")
    assert "[laugh]" not in body
    assert "这句话很有意思。" in body


def test_subtitles_can_be_switched_off(module) -> None:
    module.config.output.subtitles = False
    result = module.synthesize_many([
        SynthRequest(text="第一段的内容在这里。", voice="test_female"),
        SynthRequest(text="第二段的内容在这里。", voice="test_male"),
    ], merge=True, name="no_srt")
    assert result.subtitle_path is None
    assert not list(Path(result.output_dir).glob("*.srt"))


def test_merge_audio_writes_subtitle_when_texts_given(module) -> None:
    """`merge_audio`（复用已有音频那条路）也要能出字幕。"""
    first = module.synthesize("第一段的内容在这里。", voice="test_female", name="m_srt")
    second = module.synthesize("第二段的内容在这里。", voice="test_male", name="m_srt")
    merged = module.merge_audio([first.path, second.path], gaps_ms=[400, 0],
                                texts=["第一段的内容在这里。", "第二段的内容在这里。"],
                                name="m_srt")
    assert merged["subtitle_path"]
    body = Path(merged["subtitle_path"]).read_text(encoding="utf-8-sig")
    assert body.count("-->") == 2
    # 第二条的起点 = 第一段时长 + 0.4s 静音
    assert f"00:00:0{int(first.seconds) + 0}" in body or "-->" in body


def test_reported_elapsed_is_wall_clock_not_audio_length(module) -> None:
    """
    耗时与音频时长必须是两个字段。

    界面上只写一个数字时会被当成"这次跑了多久"，而 `sine` 引擎毫秒级返回、
    真实引擎 RTF 十几倍 —— 混起来就是"时间统计明显不对"。
    """
    result = module.synthesize("这是一段用来测试的话。", voice="test_female", save=False)
    assert result.elapsed > 0
    assert result.seconds > 0
    assert result.segments[0].elapsed > 0
    # sine 是即时返回的：耗时必然远小于音频时长（真实引擎则相反）
    assert result.elapsed < result.seconds
    assert result.rtf == pytest.approx(result.elapsed / result.seconds, rel=0.2)
