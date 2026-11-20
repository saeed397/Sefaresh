"""core/levels.py — حمایت و مقاومت افقی فقط از پیوت‌های واقعی.

نام توابع و کلیدهای خروجی نسخه اولیه حفظ شده‌اند. کمبود سطح با ATR
پر نمی‌شود؛ لیست کوتاه یا خالی یعنی شواهد کافی موجود نیست.
سطوح این فایل مرجع قیمت‌اند، نه مجوز ورود یا احتمال موفقیت معامله.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cluster_levels(prices: np.ndarray, weights: np.ndarray,
                    tolerance_pct: float) -> list[dict]:
    """خوشه‌بندی با دامنه محدود؛ بدون اتصال زنجیره‌ای سطوح دور از هم."""
    prices = np.asarray(prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if prices.ndim != 1 or weights.ndim != 1 or len(prices) != len(weights):
        raise ValueError("prices and weights must be equal-length one-dimensional arrays")
    if not np.isfinite(tolerance_pct) or tolerance_pct < 0:
        raise ValueError("tolerance_pct must be finite and non-negative")
    valid = np.isfinite(prices) & (prices > 0) & np.isfinite(weights) & (weights > 0)
    prices, weights = prices[valid], weights[valid]
    if not len(prices):
        return []
    order = np.argsort(prices, kind="stable")
    prices, weights = prices[order], weights[order]
    clusters = []
    group_prices, group_weights = [], []

    def append_group() -> None:
        clusters.append({"price": float(np.mean(group_prices)),
                         "score": float(np.sum(group_weights)),
                         "touches": len(group_prices)})

    for price, weight in zip(prices, weights):
        if group_prices and (price - group_prices[0]) / group_prices[0] * 100 > tolerance_pct:
            append_group()
            group_prices, group_weights = [], []
        group_prices.append(float(price))
        group_weights.append(float(weight))
    append_group()
    return clusters


def _pivot_prices(df: pd.DataFrame, column: str, indices: list[int]) -> np.ndarray:
    if column not in df:
        return np.array([], dtype=float)
    positions = sorted({int(i) for i in indices
                        if isinstance(i, (int, np.integer)) and not isinstance(i, bool)
                        and 0 <= i < len(df)})
    values = pd.to_numeric(df[column], errors="coerce").iloc[positions].to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def extract_levels(df: pd.DataFrame, pivot_highs: list[int], pivot_lows: list[int]) -> dict:
    """سطح major فقط خوشه با حداقل سه پیوت مستقل، طبق مرز نسخه اولیه.

    پیوت‌ها باید از find_pivots روی کندل‌های بسته همان نماد/تایم‌فریم بیایند.
    تکرار یک اندیس، برخورد تازه محسوب نمی‌شود. داده آینده اینجا خوانده نمی‌شود.
    """
    result = {"nearest_major_support": None, "nearest_major_resistance": None,
              "resistance_clusters": [], "support_clusters": [], "key_static_levels": []}
    if df.empty or "Close" not in df:
        return result
    current = pd.to_numeric(df["Close"], errors="coerce").iloc[-1]
    if not np.isfinite(current) or current <= 0:
        return result
    highs = _pivot_prices(df, "High", pivot_highs)
    lows = _pivot_prices(df, "Low", pivot_lows)
    resistance = sorted(_cluster_levels(highs, np.ones(len(highs)), .4),
                        key=lambda c: (-c["score"], c["price"]))
    support = sorted(_cluster_levels(lows, np.ones(len(lows)), .4),
                     key=lambda c: (-c["score"], c["price"]))
    strong_resistance = [c["price"] for c in resistance if c["touches"] >= 3 and c["price"] > current]
    strong_support = [c["price"] for c in support if c["touches"] >= 3 and c["price"] < current]
    result.update(nearest_major_resistance=min(strong_resistance) if strong_resistance else None,
                  nearest_major_support=max(strong_support) if strong_support else None,
                  resistance_clusters=resistance, support_clusters=support,
                  key_static_levels=sorted({c["price"] for c in resistance + support}))
    return result


def _spaced_levels(base_price: float, direction: int, atr: float, n: int,
                   min_gap_atr: float, jitter: list[float] | None = None) -> list[float]:
    """سازگاری با نسخه اولیه؛ ساخت سطح فرضی ممنوع است، پس خروجی خالی است."""
    return []


def build_target_ladder(current_price: float, atr: float, side: str,
                        level_info: dict) -> dict:
    """حداکثر سه مرجع واقعی هر گروه؛ نه سه هدف قابل معامله.

    حدسود قابل معامله در engine از سطح قوی نزدیک و دامنه تاریخی محدود می‌شود.
    مرز سه برخورد و فاصله ۰٫۶ ATR از نسخه اولیه حفظ شده است؛ فاصله‌گذاری
    بین هر دو گروه مشترک است و هرگز قیمت تازه ایجاد نمی‌کند.
    """
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    result = {"medium": [], "strong": []}
    if not np.isfinite(current_price) or current_price <= 0 or not np.isfinite(atr) or atr <= 0:
        return result
    direction = 1 if side == "BUY" else -1
    key = "resistance_clusters" if side == "BUY" else "support_clusters"
    candidates = []
    for cluster in level_info.get(key, []):
        price, touches = cluster.get("price"), cluster.get("touches")
        if not isinstance(price, (int, float, np.number)) or not np.isfinite(price) or price <= 0:
            continue
        if not isinstance(touches, (int, np.integer)) or isinstance(touches, bool) or touches < 1:
            continue
        if (price - current_price) * direction > 0:
            candidates.append((float(price), int(touches)))
    # سطح قوی در تعارض فاصله با سطح ضعیف‌تر اولویت دارد.
    candidates.sort(key=lambda item: (item[1] < 3, direction * (item[0] - current_price)))
    selected = []
    for price, touches in candidates:
        group = "strong" if touches >= 3 else "medium"
        if len(result[group]) >= 3 or any(abs(price - other) < atr * .6 for other in selected):
            continue
        selected.append(price)
        result[group].append(price)
    for values in result.values():
        values.sort(reverse=side == "SELL")
    return result