"""
音频读写与拼接 / Audio I/O and concatenation.

拼接层要做的三件事，都是「没有 SSML 时唯一可靠的节奏控制手段」：
段间插**确定长度**的静音、换角色时插更长的静音、跨引擎时统一采样率。
Inserting fixed-length silence is the one rhythm control that does not depend on
the model's behaviour.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def silence(ms: int, sample_rate: int) -> np.ndarray:
    """一段静音 / A stretch of silence."""
    return np.zeros(max(int(sample_rate * ms / 1000), 0), dtype=np.float32)


def duration_seconds(wav: np.ndarray, sample_rate: int) -> float:
    return len(wav) / sample_rate if sample_rate else 0.0


def resample(wav: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    重采样 / Resample.

    有 librosa 就用它（质量好）；没有就退回线性插值 —— **拼接不能因为少一个可选依赖
    就失败**，跨引擎混排时采样率不一致是必然会遇到的。
    Falls back to linear interpolation so joining never fails on a missing optional dep.
    """
    if src_rate == dst_rate or len(wav) == 0:
        return np.asarray(wav, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(np.asarray(wav, dtype=np.float32),
                                orig_sr=src_rate, target_sr=dst_rate).astype(np.float32)
    except ImportError:
        count = int(round(len(wav) * dst_rate / src_rate))
        source = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        target = np.linspace(0.0, 1.0, num=count, endpoint=False)
        return np.interp(target, source, np.asarray(wav, dtype=np.float32)).astype(np.float32)


def peak_normalize(wav: np.ndarray, target: float = 0.95) -> np.ndarray:
    """
    峰值归一 / Peak normalisation.

    **默认不开**：多段音频各自归一会把段间音量差抹平，听起来忽大忽小；
    只在整条拼接完之后归一才有意义。
    """
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak <= 1e-6:
        return wav
    return (wav * (target / peak)).astype(np.float32)


def join(clips: list[np.ndarray], sample_rates: list[int], gaps_ms: list[int],
         *, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    """
    拼接多段音频 / Concatenate clips with per-gap silence.

    参数 / Args:
        clips: 一维 float32 波形 / 1-D float32 waveforms.
        sample_rates: 每段的采样率（跨引擎可能不同，会统一到 `target_rate`）。
        gaps_ms: 每段**之后**的静音毫秒数，长度与 clips 相同；最后一项被忽略
            —— 整条音频结尾不该拖一段静音。
        target_rate: 统一到哪个采样率；None = 取第一段的。

    返回 / Returns:
        `(wav, sample_rate)`.

    抛出 / Raises:
        ValueError: clips 为空，或三个列表长度对不上。
    """
    if not clips:
        raise ValueError("没有可拼接的音频")
    if not (len(clips) == len(sample_rates) == len(gaps_ms)):
        raise ValueError(
            f"长度对不上：clips={len(clips)} rates={len(sample_rates)} gaps={len(gaps_ms)}")

    rate = target_rate or sample_rates[0]
    pieces: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        wav = np.asarray(clip, dtype=np.float32).reshape(-1)
        if sample_rates[index] != rate:
            wav = resample(wav, sample_rates[index], rate)
        pieces.append(wav)
        if index < len(clips) - 1 and gaps_ms[index] > 0:
            pieces.append(silence(gaps_ms[index], rate))
    return np.concatenate(pieces), rate


def write_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> Path:
    """
    落盘 / Write a WAV file. 父目录自动建。

    抛出 / Raises:
        RuntimeError: soundfile 没装（它是必需依赖，缺了说明环境不对）。
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"需要 soundfile 才能写音频：{exc}") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.asarray(wav, dtype=np.float32), sample_rate)
    return target


def to_wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """内存里的 WAV 字节 / WAV bytes in memory —— HTTP 直接返回音频用。"""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(wav, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """读一个音频文件 / Read an audio file（参考音频体检用）。"""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)          # 混成单声道 / mixed down to mono
    return np.asarray(data, dtype=np.float32), int(rate)


__all__ = [
    "duration_seconds",
    "join",
    "peak_normalize",
    "read_wav",
    "resample",
    "silence",
    "to_wav_bytes",
    "write_wav",
]
