"""
core/indicators.py
-------------------
فصل ۳ (ATR)، فصل ۴ (Pivot و خط روند)، فصل ۵ (قدرت خطوط).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=max(2, period // 2)).mean()
    return atr


def find_pivots(df: pd.DataFrame, window: int) -> tuple[list[int], list[int]]:
    """
    فصل ۴.۱: Pivot High = سقفی بالاتر از window کندل قبل و بعد از خودش.
    Pivot Low = کف پایین‌تر از window کندل قبل و بعد.
    خروجی: اندیس‌های صحیح (iloc) پیوت‌های های و لو.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    ph, pl = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window: i + window + 1]
        seg_l = lows[i - window: i + window + 1]
        if highs[i] == seg_h.max() and np.argmax(seg_h) == window:
            ph.append(i)
        if lows[i] == seg_l.min() and np.argmin(seg_l) == window:
            pl.append(i)
    return ph, pl


@dataclass
class TrendLine:
    kind: str  # "RESISTANCE" | "SUPPORT"
    slope: float
    intercept: float
    x0: int
    x1: int
    touches: int
    age_candles: int
    angle_degrees: float
    current_price: float
    distance_pct: float
    velocity: str
    strength_score: float
    strength_class: str
    touch_indices: list = field(default_factory=list)

    def value_at(self, x: int) -> float:
        return self.slope * x + self.intercept


def _fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if len(xs) < 2:
        return 0.0, float(ys[-1]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _count_touches(df: pd.DataFrame, slope: float, intercept: float,
                    kind: str, tolerance_pct: float = 0.15) -> tuple[int, list[int]]:
    n = len(df)
    xs = np.arange(n)
    line_vals = slope * xs + intercept
    price = df["High"].values if kind == "RESISTANCE" else df["Low"].values
    tol = np.abs(line_vals) * (tolerance_pct / 100.0)
    close_enough = np.abs(price - line_vals) <= np.maximum(tol, 1e-9)
    idx = list(np.where(close_enough)[0])
    return len(idx), idx


def _velocity_of_departure(df: pd.DataFrame, touch_indices: list[int]) -> str:
    """فصل ۵.۵: سرعت دورشدن قیمت پس از آخرین برخورد با خط."""
    if not touch_indices:
        return "SLOW"
    last = touch_indices[-1]
    n = len(df)
    if last >= n - 2:
        return "SLOW"
    horizon = min(5, n - 1 - last)
    close = df["Close"].values
    move = abs(close[last + horizon] - close[last])
    atr_like = df["High"].values[max(0, last - 14):last + 1].std() + 1e-9
    ratio = move / (atr_like * math.sqrt(horizon) + 1e-9)
    return "FAST" if ratio > 1.0 else "SLOW"


def _strength_score(touches: int, age_candles: int, angle: float,
                     distance_pct: float, velocity: str, n_total: int) -> tuple[float, str]:
    """
    فصل ۱.۱ (خلاصه) و فصل ۵: میانگین وزنی فاکتورها.
    وزن قوی=۳، متوسط=۲، ضعیف=۱ (طبق تعریف کاربر در سند).
    هر زیرمعیار به بازه ۰-۱۰۰ نرمال می‌شود.
    """
    # تعداد برخورد (وزن قوی=۳) — فصل ۵.۱
    touch_score = min(100.0, (touches / 6.0) * 100.0)

    # عمر/دوام (وزن قوی=۳)
    age_score = min(100.0, (age_candles / max(1, n_total)) * 130.0)

    # زاویه (وزن متوسط=۲) — زاویه > ۶۰ درجه جریمه می‌شود (فصل ۵.۲)
    angle_abs = min(abs(angle), 90.0)
    if angle_abs <= 60:
        angle_score = 100.0 - (angle_abs / 60.0) * 30.0  # زاویه معتدل بهتر
    else:
        angle_score = max(0.0, 70.0 - (angle_abs - 60.0) * 3.5)  # جریمه هیجانی/فومو

    # فاصله و فشار (وزن متوسط=۲) — فصل ۵.۴: فاصله خیلی کم فقط فشردگی نشان می‌دهد
    dist_score = max(0.0, 100.0 - abs(distance_pct) * 8.0)

    # سرعت دورشدن (وزن ضعیف=۱) — فصل ۵.۵
    vel_score = 70.0 if velocity == "FAST" else 40.0

    weights = {
        "touch": 3, "age": 3, "angle": 2, "dist": 2, "vel": 1,
    }
    total_w = sum(weights.values())
    score = (
        touch_score * weights["touch"] +
        age_score * weights["age"] +
        angle_score * weights["angle"] +
        dist_score * weights["dist"] +
        vel_score * weights["vel"]
    ) / total_w
    score = float(np.clip(score, 0, 100))

    # طبقه‌بندی طبق فصل ۵.۱ (بر اساس تعداد برخورد معتبر)
    if touches >= 4:
        cls = "STRONG"
    elif touches == 3:
        cls = "MEDIUM"
    else:
        cls = "WEAK"
    return score, cls


def build_trendline(df: pd.DataFrame, pivot_idx: list[int], kind: str) -> TrendLine | None:
    """اتصال آخرین پیوت‌های معتبر (حداکثر ۴ پیوت آخر) + رگرسیون خطی به‌عنوان جایگزین."""
    if len(pivot_idx) < 2:
        return None

    used = pivot_idx[-4:] if len(pivot_idx) > 4 else pivot_idx
    xs = np.array(used, dtype=float)
    price_col = df["High"] if kind == "RESISTANCE" else df["Low"]
    ys = price_col.values[used].astype(float)

    # روش ۱: اتصال ساختاری (دو پیوت آخر)
    x_struct = np.array([used[-2], used[-1]], dtype=float)
    y_struct = price_col.values[[used[-2], used[-1]]].astype(float)
    slope_s, intercept_s = _fit_line(x_struct, y_struct)

    # روش ۲: رگرسیون خطی روی همه پیوت‌های استفاده‌شده
    slope_r, intercept_r = _fit_line(xs, ys)

    touches_s, idx_s = _count_touches(df, slope_s, intercept_s, kind)
    touches_r, idx_r = _count_touches(df, slope_r, intercept_r, kind)

    # خط معتبرتر (برخورد بیشتر) انتخاب می‌شود — فصل ۴.۳
    if touches_r >= touches_s:
        slope, intercept, touches, touch_idx = slope_r, intercept_r, touches_r, idx_r
    else:
        slope, intercept, touches, touch_idx = slope_s, intercept_s, touches_s, idx_s

    n = len(df)
    x0, x1 = used[0], n - 1
    age_candles = x1 - used[0]
    angle_degrees = math.degrees(math.atan(slope)) if not np.isnan(slope) else 0.0

    current_price = df["Close"].iloc[-1]
    line_now = slope * (n - 1) + intercept
    distance_pct = ((current_price - line_now) / line_now) * 100.0 if line_now else 0.0

    velocity = _velocity_of_departure(df, touch_idx)
    score, cls = _strength_score(touches, age_candles, angle_degrees, distance_pct, velocity, n)

    return TrendLine(
        kind=kind, slope=slope, intercept=intercept, x0=x0, x1=x1,
        touches=touches, age_candles=age_candles, angle_degrees=angle_degrees,
        current_price=float(line_now), distance_pct=float(distance_pct),
        velocity=velocity, strength_score=score, strength_class=cls,
        touch_indices=touch_idx,
    )
