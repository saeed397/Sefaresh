"""
core/engine.py
---------------
موتور اصلی: تمام لایه‌های معماری را به ترتیب فراخوانی می‌کند:

Market Context & Data Validation -> Pivot & Trendline Construction ->
Classic Pattern Recognition -> Static S/R Detection -> Liquidity Zone
Identification -> Stop-Hunt Calibration -> Dual Signal Generation ->
Stop Loss Model -> Take Profit Model -> Risk & Confidence Filter ->
Final Decision -> (Performance Logging از سکوپ این اپ خارج است)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import indicators as ind
from . import patterns as pat
from . import levels as lv
from . import liquidity as liq
from . import signals as sig


@dataclass
class AnalysisResult:
    ok: bool
    reason: str = ""
    market: dict = field(default_factory=dict)
    upper_line: ind.TrendLine | None = None
    lower_line: ind.TrendLine | None = None
    pattern: pat.PatternResult | None = None
    level_info: dict = field(default_factory=dict)
    liquidity_zone_buy: liq.LiquidityZone | None = None
    liquidity_zone_sell: liq.LiquidityZone | None = None
    stophunt_buy: liq.StopHuntCalibration | None = None
    stophunt_sell: liq.StopHuntCalibration | None = None
    standard_signal: dict = field(default_factory=dict)
    stophunt_signal: dict = field(default_factory=dict)
    confidence: sig.ConfidenceScore | None = None
    hard_filters_passed: bool = False
    hard_filter_reasons: list = field(default_factory=list)
    final_decision: str = "HOLD"
    ladder_buy: dict = field(default_factory=dict)
    ladder_sell: dict = field(default_factory=dict)
    sides: dict = field(default_factory=dict)


def run_analysis(display_symbol: str, timeframe: str, side_preference: str,
                  rr_target: float) -> AnalysisResult:
    md = dl.load_market_data(display_symbol, timeframe)
    if not md.ok or not md.data_sufficient:
        return AnalysisResult(ok=False, reason=md.reason or "داده کافی نیست.")

    df = md.df
    cfg = dl.TIMEFRAME_MATRIX[timeframe]
    pivot_window = int(np.mean(cfg["pivot_window"]))
    atr_period = cfg["atr_period"]

    atr_series = ind.compute_atr(df, atr_period)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.dropna().empty else float(
        (df["High"] - df["Low"]).mean()
    )
    current_price = float(df["Close"].iloc[-1])

    pivot_highs, pivot_lows = ind.find_pivots(df, pivot_window)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return AnalysisResult(
            ok=False,
            reason="تعداد Pivotهای معتبر برای رسم خط روند کافی نیست؛ اطلاعات معتبر برای این بخش موجود نیست.",
        )

    upper_line = ind.build_trendline(df, pivot_highs, "RESISTANCE")
    lower_line = ind.build_trendline(df, pivot_lows, "SUPPORT")
    if upper_line is None or lower_line is None:
        return AnalysisResult(ok=False, reason="ساخت خط روند ممکن نشد.")

    # --- کارایی تاریخی الگو (فصل ۷.۵) : تخمین ساده بر مبنای تعداد لمس معتبر خطوط
    total_touches = upper_line.touches + lower_line.touches
    historical_efficiency = None
    if total_touches >= 20:
        historical_efficiency = round(min(95.0, 40 + total_touches * 1.5), 1)

    pattern = pat.analyze_pattern(df, upper_line, lower_line, historical_efficiency)

    level_info = lv.extract_levels(df, pivot_highs, pivot_lows)

    # --- ناحیه نقدینگی (فصل ۱۰) : زیر آخرین کف ماژور (خرید) و بالای آخرین سقف ماژور (فروش)
    last_low_idx = pivot_lows[-1]
    last_high_idx = pivot_highs[-1]
    last_low_price = float(df["Low"].iloc[last_low_idx])
    last_high_price = float(df["High"].iloc[last_high_idx])

    liq_zone_buy = liq.identify_liquidity_zone(
        df, last_low_price, atr_value, "BUY_SIDE", "PIVOT", lower_line.strength_class
    )
    liq_zone_sell = liq.identify_liquidity_zone(
        df, last_high_price, atr_value, "SELL_SIDE", "PIVOT", upper_line.strength_class
    )

    stophunt_buy = liq.calibrate_stop_hunt(
        df, atr_value, atr_series, pivot_lows, last_low_price, "BUY_SIDE", display_symbol
    )
    stophunt_sell = liq.calibrate_stop_hunt(
        df, atr_value, atr_series, pivot_highs, last_high_price, "SELL_SIDE", display_symbol
    )

    def build_side(side: str) -> dict:
        upper_now = upper_line.current_price
        lower_now = lower_line.current_price

        base_stop = sig.structural_stop_loss(
            side, last_low_price if side == "BUY" else last_high_price, atr_value
        )
        std_rows = sig.build_risk_ladder(side, current_price, base_stop, atr_value, rr_target)
        std_rr = std_rows[0].rr if std_rows else 0.0

        action, trigger = sig.immediate_signal(
            side, current_price, upper_line.strength_class, lower_line.strength_class,
            pattern.breakout_volume_ratio, pattern.pattern_status, pattern.maturity_pct,
            std_rr,
        )

        stophunt_cal = stophunt_buy if side == "BUY" else stophunt_sell
        liq_zone = liq_zone_buy if side == "BUY" else liq_zone_sell
        entry_sh = stophunt_cal.optimized_limit_price
        base_stop_sh = sig.anti_hunt_stop_loss(side, stophunt_cal.calibrated_boundary, atr_value)
        sh_rows = sig.build_risk_ladder(side, entry_sh, base_stop_sh, atr_value, rr_target)
        sh_rr = sh_rows[0].rr if sh_rows else 0.0
        sh_action, sh_trigger = sig.limit_signal(
            side, stophunt_cal.hypothesis_status, stophunt_cal.calibration_status, sh_rr
        )

        ladder = lv.build_target_ladder(current_price, atr_value, side, level_info)

        sr_confluence = bool(
            (side == "BUY" and level_info["nearest_major_resistance"] and
             abs(level_info["nearest_major_resistance"] - std_rows[0].take_profit) / current_price < 0.01)
            or
            (side == "SELL" and level_info["nearest_major_support"] and
             abs(level_info["nearest_major_support"] - std_rows[0].take_profit) / current_price < 0.01)
        )
        stophunt_quality = 70.0 if stophunt_cal.calibration_status == "CALIBRATED" else (
            45.0 if stophunt_cal.calibration_status == "PARTIALLY_CALIBRATED" else 25.0
        )

        conf = sig.compute_confidence(
            upper_line.strength_score, lower_line.strength_score,
            pattern.breakout_volume_ratio, std_rr, pattern.maturity_pct,
            sr_confluence, stophunt_quality,
        )

        sl_inside_liquidity = liq_zone.zone_lower <= base_stop <= liq_zone.zone_upper
        passed, reasons = sig.apply_hard_filters(
            data_sufficient=md.data_sufficient, rr=std_rr, entry_defined=True,
            stop_defined=base_stop is not None, sl_inside_liquidity=sl_inside_liquidity,
            pattern_valid=pattern.pattern_status != "UNKNOWN",
        )
        decision = sig.final_decision(action, passed, conf.score_class)

        return {
            "side": side,
            "standard": {
                "entry": current_price, "rows": std_rows, "action": action, "trigger": trigger,
                "rr": std_rr,
            },
            "stophunt": {
                "entry": entry_sh, "rows": sh_rows, "action": sh_action, "trigger": sh_trigger,
                "calibration": stophunt_cal, "liquidity_zone": liq_zone, "rr": sh_rr,
            },
            "ladder": ladder,
            "confidence": conf,
            "hard_filters_passed": passed,
            "hard_filter_reasons": reasons,
            "final_decision": decision,
        }

    sides_to_build = []
    if side_preference in ("BUY", "BOTH"):
        sides_to_build.append("BUY")
    if side_preference in ("SELL", "BOTH"):
        sides_to_build.append("SELL")

    built = {s: build_side(s) for s in sides_to_build}

    primary_side = sides_to_build[0]
    primary = built[primary_side]

    return AnalysisResult(
        ok=True,
        market={
            "symbol": display_symbol, "timeframe": timeframe, "current_price": current_price,
            "candles_analyzed": md.candles_analyzed, "atr_value": atr_value,
            "atr_period": atr_period, "pivot_window": pivot_window,
            "data_source": md.data_source, "data_sufficient": md.data_sufficient,
            "reason": md.reason, "df": df,
        },
        upper_line=upper_line, lower_line=lower_line, pattern=pattern, level_info=level_info,
        liquidity_zone_buy=liq_zone_buy, liquidity_zone_sell=liq_zone_sell,
        stophunt_buy=stophunt_buy, stophunt_sell=stophunt_sell,
        standard_signal={s: built[s]["standard"] for s in built},
        stophunt_signal={s: built[s]["stophunt"] for s in built},
        confidence=primary["confidence"],
        hard_filters_passed=primary["hard_filters_passed"],
        hard_filter_reasons=primary["hard_filter_reasons"],
        final_decision=primary["final_decision"],
        ladder_buy=built.get("BUY", {}).get("ladder", {}),
        ladder_sell=built.get("SELL", {}).get("ladder", {}),
        sides=built,
    ) if built else AnalysisResult(ok=False, reason="جهت سیگنالی انتخاب نشده است.")
