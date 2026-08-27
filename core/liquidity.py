"""
core/liquidity.py
------------------
فصل ۱۰: نقدینگی و فرضیه Stop-Hunt (نسخه اصلاح‌شده احتمالی)
فصل ۱۱: کالیبراسیون عمق Stop-Hunt

توجه: طبق فصل ۱۰.۱ نیت بازیگران بزرگ قابل مشاهده مستقیم نیست؛ خروجی این ماژول
همواره به‌صورت «فرضیه احتمالی» و با بازه (نه قیمت قطعی) بیان می‌شود.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# فصل ۱۱.۳ — بازه‌های عمومی (فقط مرجع، در صورت نبود کالیبراسیون اختصاصی)
GENERIC_MULTIPLIER_RANGE = {
    "LOW_VOL": (1.0, 1.5),   # BTC و کم‌نوسان
    "HIGH_VOL": (2.0, 3.0),  # ETH و آلت‌کوین‌ها
}

LOW_VOL_ASSETS = {"BTC"}


@dataclass
class LiquidityZone:
    side: str  # BUY_SIDE | SELL_SIDE
    structural_level: float
    zone_lower: float
    zone_upper: float
    zone_mid: float
    source: str
    strength: str


@dataclass
class StopHuntCalibration:
    hypothesis_status: str
    calibrated_boundary: float
    optimized_limit_price: float
    sweep_depth_pct: float
    backtest_volatility_multiplier: float
    calibration_status: str  # CALIBRATED | PARTIALLY_CALIBRATED | NOT_CALIBRATED
    sample_size: int
    evidence_for: list
    evidence_against: list


def identify_liquidity_zone(df: pd.DataFrame, structural_level: float, atr: float,
                             side: str, source: str, strength: str) -> LiquidityZone:
    """فصل ۱۰.۵: محدوده (نه قیمت قطعی) — مرز ابتدایی/میانی/نهایی."""
    span = max(atr * 0.35, structural_level * 0.001)
    if side == "BUY_SIDE":  # زیر کف‌ها (برای سیگنال LONG / Stop-Hunt خرید)
        zone_lower = structural_level - span * 1.6
        zone_upper = structural_level - span * 0.2
    else:  # SELL_SIDE — بالای سقف‌ها
        zone_lower = structural_level + span * 0.2
        zone_upper = structural_level + span * 1.6
    zone_mid = (zone_lower + zone_upper) / 2.0
    return LiquidityZone(
        side=side, structural_level=structural_level, zone_lower=zone_lower,
        zone_upper=zone_upper, zone_mid=zone_mid, source=source, strength=strength,
    )


def extract_backtest_multiplier(df: pd.DataFrame, atr_series: pd.Series,
                                 pivot_idx: list[int], side: str) -> tuple[float, int, str]:
    """
    فصل ۱۱.۵: الگوریتم استخراج ضریب نوسان اختصاصی از تاریخچه واقعی همان رمزارز.
    1) رخدادهای Sweep تاریخی: عبور از پیوت + بازگشت در حداکثر ۳ کندل
    2) Sweep_Ratio_i = عمق واقعی / ATR در زمان رخداد
    3) ضریب نهایی = میانه آخرین ۳۰ رخداد معتبر (کمتر از ۳۰ نمونه -> NOT_CALIBRATED)
    """
    ratios = []
    n = len(df)
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values

    for p in pivot_idx:
        if p >= n - 4:
            continue
        atr_p = atr_series.iloc[p] if p < len(atr_series) else np.nan
        if pd.isna(atr_p) or atr_p <= 0:
            continue
        level = lows[p] if side == "BUY_SIDE" else highs[p]
        window = range(p + 1, min(p + 4, n))
        for w in window:
            if side == "BUY_SIDE" and lows[w] < level:
                depth = level - lows[w]
                # بازگشت در حداکثر ۳ کندل بعدی
                recovered = any(closes[k] > level for k in range(w, min(w + 3, n)))
                if recovered:
                    ratios.append(depth / atr_p)
                break
            if side == "SELL_SIDE" and highs[w] > level:
                depth = highs[w] - level
                recovered = any(closes[k] < level for k in range(w, min(w + 3, n)))
                if recovered:
                    ratios.append(depth / atr_p)
                break

    sample = len(ratios)
    recent = ratios[-30:] if sample > 30 else ratios

    if sample >= 30:
        status = "CALIBRATED"
        multiplier = float(np.median(recent))
    elif sample >= 10:
        status = "PARTIALLY_CALIBRATED"
        multiplier = float(np.median(recent)) if recent else None
    else:
        status = "NOT_CALIBRATED"
        multiplier = None

    return multiplier, sample, status


def calibrate_stop_hunt(df: pd.DataFrame, atr_value: float, atr_series: pd.Series,
                         pivot_idx: list[int], structural_level: float, side: str,
                         display_symbol: str) -> StopHuntCalibration:
    """فصل ۱۱.۱ تا ۱۱.۴: کالیبراسیون کامل عمق Stop-Hunt."""
    multiplier, sample, status = extract_backtest_multiplier(df, atr_series, pivot_idx, side)

    if multiplier is None:
        vol_bucket = "LOW_VOL" if display_symbol in LOW_VOL_ASSETS else "HIGH_VOL"
        lo, hi = GENERIC_MULTIPLIER_RANGE[vol_bucket]
        multiplier = (lo + hi) / 2.0

    direction = -1 if side == "BUY_SIDE" else 1
    boundary = structural_level + direction * multiplier * atr_value
    optimized_limit = (structural_level + boundary) / 2.0  # نقطه بهینه ناحیه Liquidity Pool

    sweep_depth_pct = abs(structural_level - boundary) / structural_level * 100.0

    evidence_for, evidence_against = [], []
    if status == "CALIBRATED":
        evidence_for.append(f"ضریب اختصاصی از {sample} رخداد تاریخی Sweep استخراج شد.")
    elif status == "PARTIALLY_CALIBRATED":
        evidence_for.append(f"ضریب تقریبی از {sample} رخداد (کمتر از حد نصاب ۳۰) استخراج شد.")
        evidence_against.append("نمونه کافی برای کالیبراسیون کامل وجود ندارد.")
    else:
        evidence_against.append("داده کافی برای استخراج ضریب اختصاصی موجود نیست؛ از بازه عمومی مرجع استفاده شد.")

    hypothesis_status = "LIQUIDITY_ZONE_UNTESTED"

    return StopHuntCalibration(
        hypothesis_status=hypothesis_status,
        calibrated_boundary=float(boundary),
        optimized_limit_price=float(optimized_limit),
        sweep_depth_pct=float(sweep_depth_pct),
        backtest_volatility_multiplier=float(multiplier),
        calibration_status=status,
        sample_size=sample,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
    )
