"""
core/liquidity.py
فصل ۱۰: نقدینگی و فرضیه Stop-Hunt (نسخه اصلاح‌شده)
فصل ۱۱: کالیبراسیون عمق Stop-Hunt

نیت بازیگران بزرگ مستقیم دیده نمی‌شود؛ خروجی این ماژول فقط «فرضیه احتمالی» است.
هیچ ضریب، عمق یا قیمتی بدون شواهد تاریخی همان نماد/تایم‌فریم ساخته نمی‌شود؛
در نبود شواهد کافی مقدار None (نامعلوم) برگردانده می‌شود.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_PARTIAL_SAMPLES = 10
MIN_CALIBRATED_SAMPLES = 30
RECENT_SAMPLES = 30
SWEEP_LOOKAHEAD = 3      # حداکثر کندل بعد از پیوت برای عبور از سطح
RECOVERY_WINDOW = 3      # حداکثر کندل برای بازگشت بسته‌شدن به داخل سطح


@dataclass
class LiquidityZone:
    side: str                      # BUY_SIDE | SELL_SIDE
    structural_level: float
    zone_lower: float | None
    zone_upper: float | None
    zone_mid: float | None
    source: str
    strength: str
    sample_size: int = 0
    status: str = "NOT_CALIBRATED"


@dataclass
class StopHuntCalibration:
    hypothesis_status: str
    calibrated_boundary: float | None
    optimized_limit_price: float | None
    sweep_depth_pct: float | None
    backtest_volatility_multiplier: float | None
    calibration_status: str        # CALIBRATED | PARTIALLY_CALIBRATED | NOT_CALIBRATED
    sample_size: int
    evidence_for: list = field(default_factory=list)
    evidence_against: list = field(default_factory=list)


def _valid_number(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sweep_ratios(df: pd.DataFrame, atr_series: pd.Series | None,
                  pivot_idx: list[int], side: str) -> list[float]:
    """عمق رخدادهای Sweep واقعی تقسیم بر ATR همان زمان، به ترتیب زمانی."""
    if df is None or len(df) == 0 or atr_series is None or side not in ("BUY_SIDE", "SELL_SIDE"):
        return []
    n = len(df)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    atr = pd.Series(atr_series).to_numpy(float)
    ratios, last_used = [], -1
    for p in sorted({int(i) for i in pivot_idx if 0 <= int(i) < n}):
        # پیوت تکراری یا رخداد همپوشان دوبار شمرده نمی‌شود؛ کندل باز آخر هم کنار می‌رود.
        if p <= last_used or p >= n - 1 - SWEEP_LOOKAHEAD or p >= len(atr):
            continue
        atr_p = atr[p]
        level = lows[p] if side == "BUY_SIDE" else highs[p]
        if not (_valid_number(atr_p) and atr_p > 0 and _valid_number(level)):
            continue
        for w in range(p + 1, min(p + 1 + SWEEP_LOOKAHEAD, n - 1)):
            crossed = lows[w] < level if side == "BUY_SIDE" else highs[w] > level
            if not crossed:
                continue
            end = min(w + RECOVERY_WINDOW, n - 1)
            if side == "BUY_SIDE":
                rec = [k for k in range(w, end) if closes[k] > level]
                depth = level - np.nanmin(lows[w:(rec[0] + 1) if rec else end])
            else:
                rec = [k for k in range(w, end) if closes[k] < level]
                depth = np.nanmax(highs[w:(rec[0] + 1) if rec else end]) - level
            if rec and _valid_number(depth) and depth > 0:
                ratios.append(float(depth / atr_p))
                last_used = rec[0]
            break
    return ratios


def _status(sample: int) -> str:
    if sample >= MIN_CALIBRATED_SAMPLES:
        return "CALIBRATED"
    if sample >= MIN_PARTIAL_SAMPLES:
        return "PARTIALLY_CALIBRATED"
    return "NOT_CALIBRATED"


def extract_backtest_multiplier(df: pd.DataFrame, atr_series: pd.Series,
                                pivot_idx: list[int], side: str) -> tuple[float | None, int, str]:
    """
    فصل ۱۱.۵: ضریب اختصاصی = میانه آخرین ۳۰ نسبت عمق Sweep به ATR.
    کمتر از ۱۰ نمونه -> None و NOT_CALIBRATED (بدون عدد جایگزین فرضی).
    """
    ratios = _sweep_ratios(df, atr_series, pivot_idx, side)
    sample = len(ratios)
    status = _status(sample)
    if status == "NOT_CALIBRATED":
        return None, sample, status
    return float(np.median(ratios[-RECENT_SAMPLES:])), sample, status


def identify_liquidity_zone(df: pd.DataFrame, structural_level: float, atr: float,
                            side: str, source: str, strength: str,
                            atr_series: pd.Series | None = None,
                            pivot_idx: list[int] | None = None) -> LiquidityZone:
    """
    فصل ۱۰.۵: محدوده (نه قیمت قطعی). عرض محدوده از صدک ۲۵ تا ۹۰ عمق Sweep های
    تاریخی ساخته می‌شود. بدون شواهد کافی، مرزها نامعلوم (None) می‌مانند.
    """
    ratios = _sweep_ratios(df, atr_series, pivot_idx or [], side) if atr_series is not None else []
    status = _status(len(ratios))
    lower = upper = mid = None
    if status != "NOT_CALIBRATED" and _valid_number(atr) and atr > 0 and _valid_number(structural_level):
        near, far = (float(np.quantile(ratios, q)) * atr for q in (0.25, 0.90))
        if side == "BUY_SIDE":
            lower, upper = structural_level - far, structural_level - near
        else:
            lower, upper = structural_level + near, structural_level + far
        mid = (lower + upper) / 2.0
    return LiquidityZone(side=side, structural_level=float(structural_level), zone_lower=lower,
                         zone_upper=upper, zone_mid=mid, source=source, strength=strength,
                         sample_size=len(ratios), status=status)


def calibrate_stop_hunt(df: pd.DataFrame, atr_value: float, atr_series: pd.Series,
                        pivot_idx: list[int], structural_level: float, side: str,
                        display_symbol: str) -> StopHuntCalibration:
    """فصل ۱۱.۱ تا ۱۱.۴: کالیبراسیون عمق Stop-Hunt فقط از تاریخچه همین نماد."""
    multiplier, sample, status = extract_backtest_multiplier(df, atr_series, pivot_idx, side)
    evidence_for, evidence_against = [], []
    boundary = limit = depth_pct = None
    usable = (multiplier is not None and _valid_number(atr_value) and atr_value > 0
              and _valid_number(structural_level) and structural_level > 0)
    if usable:
        direction = -1 if side == "BUY_SIDE" else 1
        boundary = float(structural_level + direction * multiplier * atr_value)
        limit = float((structural_level + boundary) / 2.0)
        depth_pct = float(abs(structural_level - boundary) / structural_level * 100.0)
    if status == "CALIBRATED":
        evidence_for.append(f"ضریب اختصاصی {display_symbol} از {sample} رخداد تاریخی Sweep استخراج شد.")
    elif status == "PARTIALLY_CALIBRATED":
        evidence_for.append(f"ضریب تقریبی از {sample} رخداد (کمتر از حد نصاب {MIN_CALIBRATED_SAMPLES}) استخراج شد.")
        evidence_against.append("نمونه کافی برای کالیبراسیون کامل وجود ندارد.")
    else:
        evidence_against.append(f"فقط {sample} رخداد Sweep تاریخی یافت شد؛ عمق شکار نامعلوم است و عدد فرضی جایگزین نشد.")
    return StopHuntCalibration(
        hypothesis_status="LIQUIDITY_ZONE_UNTESTED",
        calibrated_boundary=boundary, optimized_limit_price=limit, sweep_depth_pct=depth_pct,
        backtest_volatility_multiplier=multiplier, calibration_status=status, sample_size=sample,
        evidence_for=evidence_for, evidence_against=evidence_against)
