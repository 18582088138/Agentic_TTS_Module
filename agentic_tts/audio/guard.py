"""
失控兜底 / Runaway guard —— 本模块最重要的一段"没有它就会把噪声发出去"的代码。

────────────────────────────────────────────────────────────────────────────
问题的形状（实测，见 docs/00_research.md §2.1）
────────────────────────────────────────────────────────────────────────────
模型会**停不下来**，一直生成到 `max_new_tokens`，产出几十秒噪声或静音，
**而且不报错**。有一条 41 秒的噪声：波形指标一条问题都没报，ASR 只回读出「嗯 k」。

所以：
· **主判据只能是「时长撞上限」** —— 静音检测、能量、波形统计都测不出来；
· 辅助判据是「比预期时长长太多」（中文约 5 字/秒），它不依赖 `max_new_tokens` 的取值；
· 处置是**换种子重试**，而不是关采样 —— 实测同样 12 条文本，贪心解码 7 条失控、
  官方采样默认值 0 条失控。为了"稳定"去设 `do_sample=False` 会让情况更糟。
Retry with a different seed; never switch to greedy decoding "for stability".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RunawayVerdict:
    """判定结果 / The verdict."""

    runaway: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.runaway


def expected_seconds(text: str, chars_per_sec: float = 5.0) -> float:
    """按字数估的预期时长 / Expected duration from character count（中文约 5 字/秒）。"""
    return max(len(text) / max(chars_per_sec, 0.1), 0.2)


def check_runaway(
    seconds: float,
    text: str,
    *,
    cap_seconds: Optional[float],
    cap_ratio: float = 0.95,
    chars_per_sec: float = 5.0,
    duration_ratio: float = 3.0,
) -> RunawayVerdict:
    """
    判断一段音频是不是失控产物 / Decide whether one clip is a runaway.

    参数 / Args:
        seconds: 实际时长 / measured duration.
        text: 这一段的文本 / the text for this segment.
        cap_seconds: 时长硬上限（`max_new_tokens / codes_per_second`）；
            None 表示引擎给不出，此时只用启发式判据。
        cap_ratio: 达到上限的多少比例就算撞上限。
        chars_per_sec: 语速估计。
        duration_ratio: 超出预期时长多少倍算可疑。

    返回 / Returns:
        `RunawayVerdict`，`reason` 里写清是哪条判据命中的 ——
        事后看日志时「为什么重试了」必须一眼看出来。
    """
    if cap_seconds and seconds >= cap_seconds * cap_ratio:
        return RunawayVerdict(True, f"时长 {seconds:.1f}s 撞上限（cap {cap_seconds:.1f}s）：模型没能自己停下来")

    expected = expected_seconds(text, chars_per_sec)
    if seconds > expected * duration_ratio:
        return RunawayVerdict(
            True,
            f"时长 {seconds:.1f}s 超出预期 {expected:.1f}s 的 {duration_ratio:g} 倍（{len(text)} 字）")

    # 极短输出也是失败的一种：模型直接吐了个 EOS，音频里什么都没有。
    # 不放进上面两条是因为它的成因不同（不是"停不下来"而是"没开口"），reason 要分开写。
    if seconds < 0.15 and len(text) >= 4:
        return RunawayVerdict(True, f"时长仅 {seconds:.2f}s，{len(text)} 字的文本不可能这么短：模型没开口")

    return RunawayVerdict(False)


def next_seed(seed: int, attempt: int) -> int:
    """
    重试用的种子 / The seed for a retry.

    偏移量取质数：`seed + attempt` 在多段合成时会与**下一段**的基准种子撞上
    （段种子是 `base + index`），于是"重试"和"换段"用到同一个种子，
    看日志时分不清哪次是哪次。
    A prime offset keeps retry seeds from colliding with the next segment's seed.
    """
    return seed + 7919 * attempt


__all__ = ["RunawayVerdict", "check_runaway", "expected_seconds", "next_seed"]
