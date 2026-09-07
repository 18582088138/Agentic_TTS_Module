"""
数字、单位与日期的中文读法 / Numbers, units and dates in spoken Chinese.

────────────────────────────────────────────────────────────────────────────
为什么要自己做，而不是指望模型
────────────────────────────────────────────────────────────────────────────
模型确实认识阿拉伯数字，但**读法是有歧义的，而歧义处它只能猜**：

    2026        二零二六（年份）  还是  两千零二十六（数量）？
    4060        四零六零（型号）  还是  四千零六十（数量）？
    3.5         三点五（小数）    还是  三点五版（版本号）？
    1.7B        一点七 B         还是  十七亿？

**歧义只有上下文能解，而上下文只有我们有。** 所以这一层不是「修模型的错」，
是「把模型没有的信息补上」。
The ambiguity is resolvable only from context that the caller has and the model does not.

全部是纯函数。
"""

from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "亿", "万亿"]

# 常见单位 → 中文读法。**只收「读法与写法不同」的**，
# 像 `GB` `MB` 这种直接读字母的不收 —— 交给 `abbreviations.py`。
UNIT_READINGS: dict[str, str] = {
    "ms": "毫秒",
    "s": "秒",
    "min": "分钟",
    "h": "小时",
    "hz": "赫兹",
    "khz": "千赫兹",
    "%": "百分之",
    "°c": "摄氏度",
    "km": "千米",
    "cm": "厘米",
    "mm": "毫米",
    "kg": "千克",
}

# 数量级后缀。`304B 参数` 的 B 是 billion，不是字母 B。
MAGNITUDE_SUFFIXES: dict[str, int] = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def integer_to_chinese(value: int) -> str:
    """
    整数 → 中文读法 / Integer to spoken Chinese.

    按数量读，不是按位读：`105` → `一百零五`，不是 `一零五`。
    要按位读（年份、型号）用 `digits_to_chinese`。

    参数 / Args:
        value: 整数，支持负数 / an integer; negatives supported.

    返回 / Returns:
        中文读法 / the spoken form.

    示例 / Example::

        >>> integer_to_chinese(105)
        '一百零五'
        >>> integer_to_chinese(10000)
        '一万'
        >>> integer_to_chinese(304_000_000_000)
        '三千零四十亿'
    """
    if value < 0:
        return "负" + integer_to_chinese(-value)
    if value == 0:
        return _DIGITS[0]

    # 按四位一组切开，每组转成「千百十个」，再接上万/亿
    groups: list[int] = []
    while value:
        groups.append(value % 10_000)
        value //= 10_000

    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            # 中间的全零组产生一个「零」，但不能连续出现
            if parts and not parts[-1].endswith(_DIGITS[0]):
                parts.append(_DIGITS[0])
            continue
        chunk = _four_digits_to_chinese(group)
        # 非最高位的组不足四位时要补「零」：一亿零五百 而不是 一亿五百
        if parts and group < 1000:
            chunk = _DIGITS[0] + chunk
        parts.append(chunk + _BIG_UNITS[index])

    text = "".join(parts).rstrip(_DIGITS[0])
    # 「一十」在句中读「十」：十五，不是一十五。但「一百一十」要保留。
    if text.startswith("一十"):
        text = text[1:]
    return text or _DIGITS[0]


def _four_digits_to_chinese(value: int) -> str:
    """0–9999 → 中文，内部用 / 0-9999 to Chinese; internal."""
    result = ""
    seen_nonzero = False
    for position in range(3, -1, -1):
        digit = (value // 10**position) % 10
        if digit:
            result += _DIGITS[digit] + _UNITS[position]
            seen_nonzero = True
        elif seen_nonzero and not result.endswith(_DIGITS[0]):
            result += _DIGITS[0]
    return result.rstrip(_DIGITS[0])


def digits_to_chinese(text: str) -> str:
    """
    逐位读 / Read digit by digit.

    年份、型号、编号用这个：`2026` → `二零二六`，`4060` → `四零六零`。
    Used for years, model numbers and identifiers.
    """
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in text)


def decimal_to_chinese(text: str) -> str:
    """
    小数 → 中文 / Decimal to Chinese.

    **整数部分按数量读，小数部分逐位读** —— 这是中文的读法：
    `0.77` → `零点七七`（不是 `零点七十七`）。

    示例 / Example::

        >>> decimal_to_chinese("1.56")
        '一点五六'
        >>> decimal_to_chinese("0.77")
        '零点七七'
    """
    negative = text.startswith("-")
    body = text.lstrip("-+")
    if "." not in body:
        head = integer_to_chinese(int(body)) if body.isdigit() else body
        return ("负" if negative else "") + head

    whole, _, frac = body.partition(".")
    whole_cn = integer_to_chinese(int(whole)) if whole.isdigit() else whole
    frac_cn = digits_to_chinese(frac)
    return ("负" if negative else "") + f"{whole_cn}点{frac_cn}"


# 百分比：12.5% / 50.6%
_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
# 带数量级后缀：304B / 1.7B / 167K
_MAGNITUDE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*([KMB])\b")
# 带单位：97ms / 24kHz
_WITH_UNIT = re.compile(
    r"(?i)(-?\d+(?:\.\d+)?)\s*(ms|s|min|h|khz|hz|km|cm|mm|kg)\b"
)
# ISO 日期：2026-01-22 / 2026/01/22
_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
# 年份：`2026 年`。四位数是歧义区（年份 / 型号 / 数量），光靠位数判不了，
# 所以用上下文 —— 跟着「年」的必然是年份，逐位读。
_YEAR = re.compile(r"\b(\d{4})\s*年")
# 型号编号：`RTX 4060`、`GTX1080`、`Qwen3`、`iPhone 15`。
# 紧跟在拉丁字母标识符后面的数字是编号，不是数量，逐位读。
# 已实测踩到：`RTX 4060` 会被读成「四千零六十」。
_MODEL_NUMBER = re.compile(r"(?<![\d.])\b([A-Za-z]{2,})[ \-]?(\d{2,})\b")
# 版本号：v3.5 / 4.57.3 —— **必须在小数之前匹配**，否则 4.57.3 会被当成小数
_VERSION = re.compile(r"(?i)\bv?(\d+(?:\.\d+){2,})\b")
_V_PREFIXED = re.compile(r"(?i)\bv(\d+(?:\.\d+)?)\b")
# 纯小数与整数（最后兜底）。
#
# ⚠️ **只看前面，不看后面。** 字母在数字前的是标识符，要保护；
# 数字在字母前的是「数量 + 单位」，要转换：
#
#     Qwen3-TTS   字母在前 → 标识符 → 不碰（整体交给 lexicon.spell_identifier）
#     Q8_K_XL     字母在前 → 标识符 → 不碰
#     167GB       数字在前 → 数量   → 转成「一百六十七」，GB 留给缩写层
#
# 两个坑都实测踩过：
#   加断言之前 —— `Qwen3-TTS` 被改成 `Qwen三-TTS`
#   加了后向断言 —— `167GB` 被咬成 `十六7GB`（整体匹配被挡住，前缀却放行了）
# Only a lookbehind: letters before digits mean an identifier, digits before
# letters mean a quantity with a unit. A lookahead broke the latter.
_NUMBER = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")


def normalise_percent(text: str) -> str:
    """`12.5%` → `百分之十二点五`。中文里「百分之」在前，所以不能简单替换成后缀。"""
    return _PERCENT.sub(lambda m: "百分之" + decimal_to_chinese(m.group(1)), text)


def normalise_magnitude(text: str) -> str:
    """
    `304B` → `三千零四十亿`，`1.7B` → `十七亿`。

    ⚠️ 这一步**有歧义风险**：`Qwen3-TTS-1.7B` 里的 B 是参数量，
    但 `Plan B` 里的 B 不是。所以只匹配「数字紧跟 K/M/B」的形式，
    且要求前面确实是数字。拿不准的场景建议先跑 `04_text_robustness.yaml` 看效果。
    """

    def convert(match: re.Match[str]) -> str:
        number, suffix = match.group(1), match.group(2).lower()
        value = float(number) * MAGNITUDE_SUFFIXES[suffix]
        if value == int(value):
            return integer_to_chinese(int(value))
        return decimal_to_chinese(str(value))

    return _MAGNITUDE.sub(convert, text)


def normalise_units(text: str) -> str:
    """`97ms` → `九十七毫秒`，`24kHz` → `二十四千赫兹`。"""

    def convert(match: re.Match[str]) -> str:
        number, unit = match.group(1), match.group(2).lower()
        return decimal_to_chinese(number) + UNIT_READINGS.get(unit, unit)

    return _WITH_UNIT.sub(convert, text)


def normalise_dates(text: str) -> str:
    """
    `2026-01-22` → `二零二六年一月二十二日`。

    **年份逐位读、月日按数量读** —— 这是中文的习惯。
    """

    def convert(match: re.Match[str]) -> str:
        year, month, day = match.groups()
        return (
            f"{digits_to_chinese(year)}年"
            f"{integer_to_chinese(int(month))}月"
            f"{integer_to_chinese(int(day))}日"
        )

    return _DATE.sub(convert, text)


def normalise_years(text: str) -> str:
    """
    `2026 年` → `二零二六年`。

    四位数是**歧义区**：年份、型号、数量三种都可能，光靠位数判不了。
    这里用上下文 —— 跟着「年」的必然是年份，逐位读。
    Four-digit numbers are ambiguous; the trailing 年 resolves it.
    """
    return _YEAR.sub(lambda m: digits_to_chinese(m.group(1)) + "年", text)


def normalise_model_numbers(text: str) -> str:
    """
    `RTX 4060` → `RTX 四零六零`，`GTX1080` → `GTX 一零八零`。

    紧跟拉丁字母标识符的数字是**编号，不是数量**，必须逐位读。
    已实测踩到：不做这一步，`RTX 4060` 会被读成「四千零六十」。
    Digits attached to a Latin identifier are an identifier, not a quantity.

    ⚠️ 字母部分原样留着，交给 `abbreviations.py` 决定怎么读。
    """
    return _MODEL_NUMBER.sub(
        lambda m: f"{m.group(1)} {digits_to_chinese(m.group(2))}", text
    )


def normalise_versions(text: str) -> str:
    """
    `v3.5` → `三点五版`，`4.57.3` → `四点五七点三`。

    ⚠️ **必须在 `normalise_numbers` 之前调用**，否则 `4.57.3` 会被小数规则
    咬掉一半变成 `四点五七.3`。这是这一层里唯一有顺序依赖的一对。
    Must run before the generic number rule, which would otherwise mangle it.
    """

    def convert_multi(match: re.Match[str]) -> str:
        parts = match.group(1).split(".")
        body = "点".join(digits_to_chinese(p) if len(p) > 1 else _DIGITS[int(p)]
                         for p in parts)
        return body + "版" if match.group(0).lower().startswith("v") else body

    def convert_v(match: re.Match[str]) -> str:
        return decimal_to_chinese(match.group(1)) + "版"

    text = _VERSION.sub(convert_multi, text)
    return _V_PREFIXED.sub(convert_v, text)


def normalise_numbers(text: str, *, digit_by_digit_threshold: int = 10_000) -> str:
    """
    剩下的裸数字 / Whatever numbers are left.

    参数 / Args:
        digit_by_digit_threshold: 大于等于这个值的**整数**改成逐位读。
            默认 10000 —— 四位以上的整数多半是年份或编号（`2026`、`4060`），
            按数量读成「两千零二十六」几乎总是错的。
            Integers at or above this are read digit by digit, because four-digit
            integers are usually years or identifiers rather than quantities.

    ⚠️ 调用顺序：`normalise_versions` → `normalise_dates` → `normalise_percent`
    → `normalise_magnitude` → `normalise_units` → **`normalise_numbers`（最后）**。
    `normalise_all` 已经按这个顺序编排好了。
    """

    def convert(match: re.Match[str]) -> str:
        raw = match.group(0)
        if "." in raw:
            return decimal_to_chinese(raw)
        value = int(raw)
        if abs(value) >= digit_by_digit_threshold:
            return digits_to_chinese(raw)
        return integer_to_chinese(value)

    return _NUMBER.sub(convert, text)


def normalise_all(text: str, *, digit_by_digit_threshold: int = 10_000) -> str:
    """
    按正确顺序跑完全部数字规则 / Run every numeric rule in the correct order.

    **顺序是有依赖的**，别自己拼。三条关键依赖：

    1. 版本号必须在小数之前 —— 否则 `4.57.3` 被小数规则咬成 `四点五七.3`
    2. 日期必须在年份之前 —— 否则 `2026-01-22` 的 `2026` 先被年份规则吃掉
    3. **型号编号与年份必须在通用数字规则之前** —— 否则 `RTX 4060` 和
       `2026 年` 会被按数量读成「四千零六十」「二千零二十六」

    The order matters; each of these three has been observed to break when reordered.
    """
    text = normalise_versions(text)
    text = normalise_dates(text)
    text = normalise_years(text)
    text = normalise_model_numbers(text)
    text = normalise_percent(text)
    text = normalise_magnitude(text)
    text = normalise_units(text)
    return normalise_numbers(text, digit_by_digit_threshold=digit_by_digit_threshold)


__all__ = [
    "MAGNITUDE_SUFFIXES",
    "UNIT_READINGS",
    "decimal_to_chinese",
    "digits_to_chinese",
    "integer_to_chinese",
    "normalise_all",
    "normalise_dates",
    "normalise_magnitude",
    "normalise_model_numbers",
    "normalise_numbers",
    "normalise_percent",
    "normalise_units",
    "normalise_versions",
    "normalise_years",
]
