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
    """میانگین دامنه واقعی؛ بدون جایگزین‌کردن داده گمشده یا دوره ناقص."""
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError("period must be a positive integer")
    high, low, close = (pd.to_numeric(df[col], errors="coerce")
                        for col in ("High", "Low", "Close"))
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    valid = (np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
             & (high >= low) & (close >= low) & (close <= high))
    previous_valid = valid.shift(1, fill_value=False)
    if len(previous_valid):
        previous_valid.iloc[0] = True
    tr = tr.where(valid & previous_valid)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def find_pivots(df: pd.DataFrame, window: int) -> tuple[list[int], list[int]]:
    """
    فصل ۴.۱: Pivot High = سقفی بالاتر از window کندل قبل و بعد از خودش.
    Pivot Low = کف پایین‌تر از window کندل قبل و بعد.
    خروجی: اندیس‌های صحیح (iloc) پیوت‌های های و لو.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    ph, pl = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window: i + window + 1]
        seg_l = lows[i - window: i + window + 1]
        neighbors_h = np.delete(seg_h, window)
        neighbors_l = np.delete(seg_l, window)
        if np.isfinite(seg_h).all() and np.all(highs[i] > neighbors_h):
            ph.append(i)
        if np.isfinite(seg_l).all() and np.all(lows[i] < neighbors_l):
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
    current_price: float  # مقدار فعلی خط، نه Close بازار؛ قرارداد patterns.py
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
                    kind: str, tolerance_pct: float = 0.15,
                    start_index: int = 0, pivot_indices: list[int] | None = None) -> tuple[int, list[int]]:
    n = len(df)
    xs = np.arange(n)
    line_vals = slope * xs + intercept
    price = df["High"].values if kind == "RESISTANCE" else df["Low"].values
    tol = np.abs(line_vals) * (tolerance_pct / 100.0)
    close_enough = np.isfinite(price) & np.isfinite(line_vals) & (np.abs(price - line_vals) <= tol)
    eligible = np.zeros(n, dtype=bool)
    if pivot_indices is None:
        eligible[max(0, start_index):] = True
    else:
        for i in pivot_indices:
            if start_index <= i < n:
                eligible[i] = True
    idx = [int(i) for i in np.where(close_enough & eligible)[0]]
    # کندل‌های چسبیده به هم یک برخورد مستقل محسوب می‌شوند.
    idx = [i for j, i in enumerate(idx) if j == 0 or i > idx[j - 1] + 1]
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
    atr_like = compute_atr(df.iloc[:last + 1], min(14, last + 1)).iloc[-1]
    if not np.isfinite(atr_like) or atr_like <= 0 or not np.isfinite(move):
        return "SLOW"
    ratio = move / (atr_like * math.sqrt(horizon))
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
    if kind not in ("RESISTANCE", "SUPPORT"):
        raise ValueError("kind must be RESISTANCE or SUPPORT")
    if df.empty:
        return None
    price_col = pd.to_numeric(df["High"] if kind == "RESISTANCE" else df["Low"], errors="coerce")
    pivot_idx = sorted({int(i) for i in pivot_idx
                        if isinstance(i, (int, np.integer)) and not isinstance(i, bool)
                        and 0 <= i < len(df) and np.isfinite(price_col.iloc[i])})
    if len(pivot_idx) < 2:
        return None

    used = pivot_idx[-4:] if len(pivot_idx) > 4 else pivot_idx
    xs = np.array(used, dtype=float)
    ys = price_col.values[used].astype(float)

    # روش ۱: اتصال ساختاری (دو پیوت آخر)
    x_struct = np.array([used[-2], used[-1]], dtype=float)
    y_struct = price_col.values[[used[-2], used[-1]]].astype(float)
    slope_s, intercept_s = _fit_line(x_struct, y_struct)

    # روش ۲: رگرسیون خطی روی همه پیوت‌های استفاده‌شده
    slope_r, intercept_r = _fit_line(xs, ys)

    candidates = []
    for slope, intercept, start in ((slope_s, intercept_s, used[-2]),
                                    (slope_r, intercept_r, used[0])):
        touches, indices = _count_touches(df, slope, intercept, kind,
                                          start_index=start, pivot_indices=pivot_idx)
        if touches < 2:
            continue
        # برخوردهای قبل از تشکیل خط امتیاز نمی‌گیرند؛ خط شکسته معتبر نیست.
        line_values = slope * np.arange(indices[1], len(df)) + intercept
        prices = price_col.iloc[indices[1]:].to_numpy(dtype=float)
        tolerance = np.abs(line_values) * (0.15 / 100.0)
        valid_side = (prices <= line_values + tolerance if kind == "RESISTANCE"
                      else prices >= line_values - tolerance)
        if (np.isfinite(prices).all() and np.isfinite(line_values).all()
                and (line_values > 0).all() and valid_side.all()):
            candidates.append((slope, intercept, touches, indices))
    if not candidates:
        return None
    slope, intercept, touches, touch_idx = max(candidates, key=lambda item: item[2])

    n = len(df)
    x0, x1 = touch_idx[0], n - 1
    age_candles = x1 - x0
    # زاویه برحسب نوسان واقعی، مستقل از واحد قیمت دارایی است.
    atr_now = compute_atr(df, min(14, n)).iloc[-1]
    if not np.isfinite(atr_now) or atr_now <= 0:
        return None
    angle_degrees = math.degrees(math.atan(slope / atr_now))

    current_price = df["Close"].iloc[-1]
    line_now = slope * (n - 1) + intercept
    if not np.isfinite(current_price) or not np.isfinite(line_now) or line_now <= 0:
        return None
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