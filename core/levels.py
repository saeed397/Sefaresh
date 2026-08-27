"""
core/levels.py
---------------
فصل ۸: حمایت، مقاومت و اهداف قیمتی (سطوح افقی).
تولید سه سطح متوسط و سه سطح قوی برای مقاومت (Buy) یا حمایت (Sell) با فاصله
معنادار و بدون هم‌پوشانی، طبق درخواست دقیق کاربر در بخش خروجی.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cluster_levels(prices: np.ndarray, weights: np.ndarray, tolerance_pct: float) -> list[dict]:
    """پیوت‌های نزدیک به هم را در یک سطح خوشه‌بندی می‌کند و امتیاز (وزن) هر خوشه را جمع می‌زند."""
    if len(prices) == 0:
        return []
    order = np.argsort(prices)
    prices = prices[order]
    weights = weights[order]

    clusters = []
    cur_prices = [prices[0]]
    cur_weight = weights[0]
    for p, w in zip(prices[1:], weights[1:]):
        ref = np.mean(cur_prices)
        if abs(p - ref) / ref * 100.0 <= tolerance_pct:
            cur_prices.append(p)
            cur_weight += w
        else:
            clusters.append({"price": float(np.mean(cur_prices)), "score": float(cur_weight),
                              "touches": len(cur_prices)})
            cur_prices = [p]
            cur_weight = w
    clusters.append({"price": float(np.mean(cur_prices)), "score": float(cur_weight),
                      "touches": len(cur_prices)})
    return clusters


def extract_levels(df: pd.DataFrame, pivot_highs: list[int], pivot_lows: list[int]) -> dict:
    """نزدیک‌ترین حمایت/مقاومت معتبر و لیست سطوح کلیدی چندبار‌لمس‌شده را برمی‌گرداند."""
    high_prices = df["High"].values[pivot_highs] if pivot_highs else np.array([])
    low_prices = df["Low"].values[pivot_lows] if pivot_lows else np.array([])

    high_weights = np.ones_like(high_prices)
    low_weights = np.ones_like(low_prices)

    res_clusters = sorted(_cluster_levels(high_prices, high_weights, tolerance_pct=0.4),
                           key=lambda c: -c["score"])
    sup_clusters = sorted(_cluster_levels(low_prices, low_weights, tolerance_pct=0.4),
                           key=lambda c: -c["score"])

    current_price = float(df["Close"].iloc[-1])
    resistances = sorted([c["price"] for c in res_clusters if c["price"] > current_price])
    supports = sorted([c["price"] for c in sup_clusters if c["price"] < current_price], reverse=True)

    nearest_resistance = resistances[0] if resistances else None
    nearest_support = supports[0] if supports else None

    return {
        "nearest_major_support": nearest_support,
        "nearest_major_resistance": nearest_resistance,
        "resistance_clusters": res_clusters,
        "support_clusters": sup_clusters,
        "key_static_levels": sorted(
            [c["price"] for c in res_clusters] + [c["price"] for c in sup_clusters]
        ),
    }


def _spaced_levels(base_price: float, direction: int, atr: float, n: int,
                    min_gap_atr: float, jitter: list[float] | None = None) -> list[float]:
    """
    n سطح در جهت مشخص (۱=بالا برای مقاومت، -۱=پایین برای حمایت) با فاصله حداقلی
    min_gap_atr برابر ATR می‌سازد تا هم‌پوشانی نداشته باشند.
    """
    levels = []
    price = base_price
    for i in range(n):
        step_atr = min_gap_atr * (1 + i * 0.65)
        offset = direction * atr * step_atr
        if jitter:
            offset *= (1 + jitter[i % len(jitter)])
        price = base_price + offset if i == 0 else price + direction * atr * min_gap_atr
        levels.append(price)
    return levels


def build_target_ladder(current_price: float, atr: float, side: str,
                         level_info: dict) -> dict:
    """
    خروجی مطابق درخواست کاربر:
    برای Buy: سه مقاومت متوسط + سه مقاومت قوی
    برای Sell: سه حمایت متوسط + سه حمایت قوی
    از سطوح واقعی خوشه‌بندی‌شده در صورت وجود استفاده می‌شود و در صورت کمبود،
    با فاصله‌گذاری مبتنی بر ATR تکمیل می‌شود تا هم‌پوشانی نداشته باشند.
    """
    direction = 1 if side == "BUY" else -1
    clusters = level_info["resistance_clusters"] if side == "BUY" else level_info["support_clusters"]
    clusters = [c for c in clusters if (c["price"] - current_price) * direction > 0]
    clusters = sorted(clusters, key=lambda c: (c["price"] - current_price) * direction)

    def enforce_spacing(prices: list[float], min_gap_atr: float) -> list[float]:
        out = []
        for p in prices:
            if not out or abs(p - out[-1]) >= atr * min_gap_atr:
                out.append(p)
        return out

    real_prices = enforce_spacing([c["price"] for c in clusters], min_gap_atr=0.6)

    medium, strong = [], []
    for c in clusters:
        touches = c["touches"]
        target_list = medium if touches < 3 else strong
        if len(target_list) < 3 and (not target_list or abs(c["price"] - target_list[-1]) >= atr * 0.6):
            target_list.append(c["price"])

    # تکمیل با سطوح مصنوعی مبتنی بر ATR در صورت کمبود (بدون هم‌پوشانی)
    def fill(levels: list[float], start_mult: float) -> list[float]:
        idx = len(levels)
        last = levels[-1] if levels else current_price
        while len(levels) < 3:
            idx += 1
            last = last + direction * atr * (start_mult + idx * 0.7)
            levels.append(last)
        return levels

    medium = fill(sorted(medium, key=lambda p: (p - current_price) * direction), 1.0)
    strong_start = max(medium) if side == "BUY" else min(medium)
    strong = fill(sorted(strong, key=lambda p: (p - current_price) * direction), 1.8)

    medium = sorted(medium, key=lambda p: (p - current_price) * direction)[:3]
    strong = sorted(strong, key=lambda p: (p - current_price) * direction)[:3]

    return {"medium": medium, "strong": strong}
