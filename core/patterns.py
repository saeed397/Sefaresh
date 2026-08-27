"""
core/patterns.py
-----------------
فصل ۶: شناسایی الگوهای کلاسیک
فصل ۷: معیارهای پیشرفته اعتبارسنجی الگو (بلوغ/Apex، حجم، تقارن، White Space، کارایی تاریخی)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import TrendLine


@dataclass
class PatternResult:
    pattern_name: str
    pattern_status: str  # CONSOLIDATING | APPROACHING_BREAKOUT | BREAKING_OUT | POST_BREAKOUT
    maturity_pct: float
    volume_trend: str  # DECREASING | FLAT | INCREASING
    breakout_volume_ratio: float
    temporal_symmetry_pct: float
    white_space: str  # YES | NO
    white_space_quality: str  # HIGH | MEDIUM | LOW
    historical_efficiency: float | None
    geometric_target: float | None
    bias: str  # BULLISH | BEARISH | NEUTRAL


def _classify_shape(upper: TrendLine, lower: TrendLine) -> tuple[str, str]:
    """فصل ۶.۳: همگرایی/موازی بودن خطوط -> نام الگو و جهت احتمالی."""
    up_slope, lo_slope = upper.slope, lower.slope
    flat_th = 0.02 * (abs(upper.current_price) + 1e-9) / max(1, upper.x1 - upper.x0)

    up_flat = abs(up_slope) < flat_th
    lo_flat = abs(lo_slope) < flat_th
    up_rising = up_slope > flat_th
    lo_rising = lo_slope > flat_th
    up_falling = up_slope < -flat_th
    lo_falling = lo_slope < -flat_th

    converging = (up_falling or up_flat) and (lo_rising or lo_flat) and not (up_flat and lo_flat)

    if up_flat and lo_flat:
        return "مستطیل (Rectangle)", "NEUTRAL"
    if up_flat and lo_rising:
        return "مثلث صعودی (Ascending Triangle)", "BULLISH"
    if lo_flat and up_falling:
        return "مثلث نزولی (Descending Triangle)", "BEARISH"
    if up_falling and lo_rising:
        return "مثلث متقارن (Symmetrical Triangle)", "NEUTRAL"
    if up_rising and lo_rising:
        return "کانال صعودی (Ascending Channel)", "BULLISH"
    if up_falling and lo_falling:
        return "کانال نزولی (Descending Channel)", "BEARISH"
    if converging:
        return "کنج (Wedge)", "NEUTRAL"
    return "بدون الگوی کامل — نزدیک‌ترین فرم در حال شکل‌گیری", "NEUTRAL"


def _volume_trend(df: pd.DataFrame, x0: int, x1: int) -> str:
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return "FLAT"
    seg = df["Volume"].iloc[x0:x1 + 1].dropna()
    if len(seg) < 4:
        return "FLAT"
    half = len(seg) // 2
    first, second = seg.iloc[:half].mean(), seg.iloc[half:].mean()
    if second < first * 0.85:
        return "DECREASING"
    if second > first * 1.15:
        return "INCREASING"
    return "FLAT"


def _breakout_volume_ratio(df: pd.DataFrame) -> float:
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return 1.0
    vol = df["Volume"].dropna()
    if len(vol) < 15:
        return 1.0
    recent_avg = vol.iloc[-15:-1].mean() or 1e-9
    last = vol.iloc[-1]
    return float(last / recent_avg)


def _temporal_symmetry(upper: TrendLine, lower: TrendLine) -> float:
    """فصل ۷.۳: تقارن زمانی بین دو خط (تلورانس مطلوب < ۲۰٪)."""
    age_u = max(1, upper.age_candles)
    age_l = max(1, lower.age_candles)
    diff_pct = abs(age_u - age_l) / max(age_u, age_l) * 100.0
    symmetry_pct = max(0.0, 100.0 - diff_pct)
    return round(symmetry_pct, 1)


def _apex_maturity(df: pd.DataFrame, upper: TrendLine, lower: TrendLine) -> float:
    """فصل ۷.۱: درصد بلوغ الگو تا نقطه Apex (تلاقی نظری دو خط)."""
    n = len(df)
    slope_diff = upper.slope - lower.slope
    if abs(slope_diff) < 1e-9:
        # خطوط تقریبا موازی (کانال/مستطیل) -> بلوغ بر اساس طول عمر نسبت به کل بازه
        age = max(upper.age_candles, lower.age_candles)
        return float(np.clip((age / max(1, n)) * 100.0, 0, 100))
    x_apex = (lower.intercept - upper.intercept) / slope_diff
    start_x = min(upper.x0, lower.x0)
    total_span = x_apex - start_x
    elapsed = (n - 1) - start_x
    if total_span <= 0:
        return 100.0
    maturity = (elapsed / total_span) * 100.0
    return float(np.clip(maturity, 0, 150))


def _white_space(df: pd.DataFrame, upper: TrendLine, lower: TrendLine) -> tuple[str, str]:
    """فصل ۷.۴: فضای خالی داخل الگو نسبت به کندل‌های واقعی."""
    x0 = min(upper.x0, lower.x0)
    x1 = len(df) - 1
    span = max(1, x1 - x0)
    density = span / max(1, (upper.touches + lower.touches))
    if density > 15:
        return "YES", "LOW"
    if density > 8:
        return "YES", "MEDIUM"
    return "NO", "HIGH"


def _pattern_status(maturity_pct: float, current_price: float,
                     upper_now: float, lower_now: float) -> str:
    if current_price > upper_now:
        return "BREAKING_OUT"
    if current_price < lower_now:
        return "BREAKING_OUT"
    if maturity_pct >= 75:
        return "APPROACHING_BREAKOUT"
    if maturity_pct >= 40:
        return "CONSOLIDATING"
    return "CONSOLIDATING"


def _geometric_target(pattern_name: str, upper: TrendLine, lower: TrendLine,
                       breakout_price: float, bias: str) -> float | None:
    """فصل ۸.۲: تارگت هندسی بر اساس نوع الگو."""
    height = abs(upper.current_price - lower.current_price)
    if height <= 0:
        return None
    direction = 1 if bias == "BULLISH" else (-1 if bias == "BEARISH" else 1)
    if "مثلث" in pattern_name or "کنج" in pattern_name:
        target = breakout_price + direction * height  # عمق مثلث از نقطه شکست
    elif "مستطیل" in pattern_name:
        target = breakout_price + direction * height  # ارتفاع از نقطه شکست
    elif "کانال" in pattern_name:
        target = breakout_price + direction * height  # عرض کانال
    else:
        target = breakout_price + direction * height
    return float(target)


def analyze_pattern(df: pd.DataFrame, upper: TrendLine, lower: TrendLine,
                     historical_efficiency: float | None) -> PatternResult:
    pattern_name, bias = _classify_shape(upper, lower)

    x0 = min(upper.x0, lower.x0)
    x1 = len(df) - 1
    maturity = _apex_maturity(df, upper, lower)
    vol_trend = _volume_trend(df, x0, x1)
    breakout_ratio = _breakout_volume_ratio(df)
    symmetry = _temporal_symmetry(upper, lower)
    white_space, ws_quality = _white_space(df, upper, lower)

    current_price = float(df["Close"].iloc[-1])
    status = _pattern_status(maturity, current_price, upper.current_price, lower.current_price)

    target = _geometric_target(pattern_name, upper, lower, current_price, bias)

    return PatternResult(
        pattern_name=pattern_name,
        pattern_status=status,
        maturity_pct=round(min(maturity, 100.0), 1),
        volume_trend=vol_trend,
        breakout_volume_ratio=round(breakout_ratio, 2),
        temporal_symmetry_pct=symmetry,
        white_space=white_space,
        white_space_quality=ws_quality,
        historical_efficiency=historical_efficiency,
        geometric_target=target,
        bias=bias,
    )
