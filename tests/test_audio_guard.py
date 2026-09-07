"""
音频拼接与失控兜底的单元测试 / Unit tests for joining and the runaway guard.

复测 / Re-run:
    python -m pytest tests/test_audio_guard.py -q

为什么失控兜底值得单测 / Why the guard needs its own tests:
实测过一条 41 秒的失控噪声 —— 波形指标一条没报、ASR 只回读出「嗯 k」。
**能测出它的只有「时长撞上限」这一条判据**，所以这条判据的边界必须钉死。
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_tts.audio.guard import check_runaway, expected_seconds, next_seed
from agentic_tts.audio.io import (
    duration_seconds,
    join,
    peak_normalize,
    read_wav,
    resample,
    silence,
    to_wav_bytes,
    write_wav,
)


# --------------------------------------------------------------- 失控 / guard


def test_hitting_the_cap_is_a_runaway() -> None:
    """主判据：撞上限就是模型没能自己停下来。"""
    verdict = check_runaway(650.0, "很短的一句话", cap_seconds=655.0)
    assert verdict and "撞上限" in verdict.reason


def test_just_below_the_cap_ratio_is_not() -> None:
    # 2000 字（按 5 字/秒预期 400s）跑出 600s：既没撞上限也在启发式判据之内
    assert not check_runaway(600.0, "很短" * 1000, cap_seconds=655.0)


def test_too_long_for_the_text_is_a_runaway_even_without_a_cap() -> None:
    """辅助判据不依赖 max_new_tokens 的取值（中文约 5 字/秒）。"""
    verdict = check_runaway(41.0, "十个字的一句话啊", cap_seconds=None)
    assert verdict and "超出预期" in verdict.reason


def test_normal_duration_passes() -> None:
    assert not check_runaway(2.0, "十个字的一句话啊", cap_seconds=655.0)


def test_silence_short_output_is_flagged_separately() -> None:
    """「没开口」与「停不下来」成因不同，reason 要分得开。"""
    verdict = check_runaway(0.05, "这是一句正常长度的话", cap_seconds=655.0)
    assert verdict and "没开口" in verdict.reason


def test_expected_seconds_scales_with_length() -> None:
    assert expected_seconds("十个字的一句话啊") == pytest.approx(8 / 5.0)


def test_retry_seed_does_not_collide_with_next_segment() -> None:
    """
    段种子是 `base + index`，重试种子必须不落在下一段头上，
    否则日志里分不清哪次是重试、哪次是换段。
    """
    base = 20260907
    assert next_seed(base, 1) != base + 1
    assert next_seed(base, 1) != next_seed(base + 1, 0)


# ---------------------------------------------------------------- 拼接 / join


def _tone(seconds: float, rate: int = 24000) -> np.ndarray:
    return np.sin(np.arange(int(seconds * rate), dtype=np.float32) / rate * 440).astype(np.float32)


def test_join_inserts_gaps_between_but_not_after() -> None:
    """结尾不该拖一段静音 —— 最后一项的 gap 必须被忽略。"""
    clips = [_tone(1.0), _tone(1.0)]
    wav, rate = join(clips, [24000, 24000], [500, 500])
    assert rate == 24000
    assert duration_seconds(wav, rate) == pytest.approx(2.5, abs=0.01)


def test_join_unifies_sample_rates() -> None:
    """跨引擎混排时采样率不一致是必然会遇到的，拼接前必须统一。"""
    wav, rate = join([_tone(1.0, 24000), _tone(1.0, 16000)], [24000, 16000], [0, 0],
                     target_rate=24000)
    assert rate == 24000
    assert duration_seconds(wav, rate) == pytest.approx(2.0, abs=0.02)


def test_join_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="长度对不上"):
        join([_tone(0.1)], [24000], [0, 0])


def test_join_rejects_empty() -> None:
    with pytest.raises(ValueError, match="没有可拼接"):
        join([], [], [])


def test_resample_falls_back_without_librosa(monkeypatch) -> None:
    """少一个可选依赖不该让拼接失败。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = resample(_tone(1.0, 24000), 24000, 16000)
    assert len(out) == pytest.approx(16000, abs=2)


def test_peak_normalize_leaves_silence_alone() -> None:
    quiet = silence(100, 24000)
    assert np.array_equal(peak_normalize(quiet), quiet)


def test_write_and_read_roundtrip(tmp_path) -> None:
    path = write_wav(tmp_path / "sub" / "a.wav", _tone(0.5), 24000)
    assert path.is_file()
    wav, rate = read_wav(path)
    assert rate == 24000
    assert duration_seconds(wav, rate) == pytest.approx(0.5, abs=0.01)


def test_wav_bytes_have_a_riff_header() -> None:
    assert to_wav_bytes(_tone(0.1), 24000)[:4] == b"RIFF"
