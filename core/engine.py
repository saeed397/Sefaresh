"""core/engine.py — موتور اصلاح‌شده با توقف ایمن و شواهد قابل ردیابی.

رابط run_analysis و فیلدهای قبلی حفظ شده‌اند. چهار ورودی اولیه بدون تغییرند؛
پارامترهای pivot_sensitivity و require_htf_confirmation اختیاری هستند.
rr_target حداقل نسبت قابل قبول است، نه ضریب ساخت حدسود.
قیمت‌ها فقط از OHLCV و نمونه‌های تاریخی همان نماد/تایم‌فریم ساخته می‌شوند.
برای اتصال کامل صفحه و توضیحات، نسخه‌های بعدی app و explanations لازم‌اند.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import indicators as ind
from . import patterns as pat
from . import levels as lv
from . import liquidity as liq
from . import signals as sig

# تنظیمات محافظه‌کارانه موتور؛ احتمال موفقیت از این عددها ساخته نمی‌شود.
MIN_SAMPLES = 5
EVENT_WINDOW = 12
OUTCOME_WINDOW = 12
PHASE_NAMES = ("Sweep", "Reclaim", "Displacement", "CHoCH")


@dataclass
class AnalysisResult:
    ok: bool
    reason: str = ""
    market: dict = field(default_factory=dict)
    upper_line: Any = None
    lower_line: Any = None
    pattern: Any = None
    level_info: dict = field(default_factory=dict)
    liquidity_zone_buy: Any = None
    liquidity_zone_sell: Any = None
    stophunt_buy: Any = None
    stophunt_sell: Any = None
    standard_signal: dict = field(default_factory=dict)
    stophunt_signal: dict = field(default_factory=dict)
    confidence: Any = None
    hard_filters_passed: bool = False
    hard_filter_reasons: list = field(default_factory=list)
    final_decision: str = "HOLD"
    ladder_buy: dict = field(default_factory=dict)
    ladder_sell: dict = field(default_factory=dict)
    sides: dict = field(default_factory=dict)
    liquidity_sweep_signal: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    confirmation: dict = field(default_factory=dict)


@dataclass
class RiskRow:
    risk_percent: int
    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    success_probability: float
    sample_count: int
    risk_label: str = "احتمال تاریخی؛ نه درصد سرمایه در معرض زیان"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _valid_frame(df: Any) -> bool:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if not set(columns).issubset(df.columns):
        return False
    try:
        values = df[columns].to_numpy(dtype=float)
        return bool(np.isfinite(values).all() and (values[:, :4] > 0).all()
                    and (values[:, 4] >= 0).all()
                    and (df.High >= df[["Open", "Close", "Low"]].max(axis=1)).all()
                    and (df.Low <= df[["Open", "Close", "High"]].min(axis=1)).all()
                    and isinstance(df.index, pd.DatetimeIndex)
                    and df.index.is_monotonic_increasing and df.index.is_unique)
    except (ValueError, TypeError):
        return False


def _window(cfg: dict, sensitivity: Any) -> int:
    low, high = cfg["pivot_window"]
    if sensitivity is None:
        return max(2, int(round((low + high) / 2)))
    if isinstance(sensitivity, str):
        named = {"HIGH": low, "MEDIUM": (low + high) / 2, "LOW": high,
                 "زیاد": low, "متوسط": (low + high) / 2, "کم": high}
        if sensitivity not in named:
            raise ValueError("حساسیت پیوت معتبر نیست.")
        return max(2, int(round(named[sensitivity])))
    number = _number(sensitivity)
    if number is None or number < 2 or number != int(number):
        raise ValueError("پنجره پیوت باید عدد صحیح و حداقل ۲ باشد.")
    return int(number)


def _events(df: pd.DataFrame, atr: pd.Series, side: str, lookback: int) -> list[dict]:
    """شکار سطوح محلی؛ هر چهار مرحله روی کندل‌های بسته و متفاوت، به ترتیب.

    سطوح هر رویداد فقط از کندل‌های قبل از شکار محاسبه می‌شوند.
    پنجره نتیجه از کندل بعد از CHoCH آغاز می‌شود؛ رویدادهای همپوشان شمرده نمی‌شوند.
    """
    events: list[dict] = []
    next_start = lookback
    buy = side == "BUY"
    for sweep in range(lookback, len(df)):
        if sweep < next_start:
            continue
        a = _number(atr.iloc[sweep - 1])
        if a is None or a <= 0:
            continue
        prior = df.iloc[sweep - lookback:sweep]
        level = float(prior.Low.min() if buy else prior.High.max())
        structure = float(prior.High.max() if buy else prior.Low.min())
        extreme = float(df.Low.iloc[sweep] if buy else df.High.iloc[sweep])
        if not (extreme < level if buy else extreme > level):
            continue
        phases = {"Sweep": sweep, "Reclaim": None, "Displacement": None, "CHoCH": None}
        limit = min(len(df), sweep + EVENT_WINDOW + 1)
        for j in range(sweep + 1, limit):
            close, opening = float(df.Close.iloc[j]), float(df.Open.iloc[j])
            if phases["Reclaim"] is None:
                if close > level if buy else close < level:
                    phases["Reclaim"] = j
                continue
            if phases["Displacement"] is None:
                body = close - opening if buy else opening - close
                if body >= a and (close > level if buy else close < level):
                    phases["Displacement"] = j
                continue
            if close > structure if buy else close < structure:
                phases["CHoCH"] = j
                break
        end = phases["CHoCH"]
        # العمق يشمل كل الحركة حتى اكتمال المراحل، وليس أول شمعة وحدها.
        stop_index = end if end is not None else limit - 1
        segment = df.iloc[sweep:stop_index + 1]
        extreme = float(segment.Low.min() if buy else segment.High.max())
        event = {"phases": phases, "complete": end is not None, "level": level,
                 "structure": structure, "extreme": extreme, "atr": a,
                 "depth_atr": abs(level - extreme) / a, "sweep_index": sweep}
        if end is not None:
            event["entry"] = float(df.Close.iloc[end])
            future = df.iloc[end + 1:end + 1 + OUTCOME_WINDOW]
            event["resolved"] = len(future) == OUTCOME_WINDOW
            event["future"] = future
            next_start = end + OUTCOME_WINDOW + 1
        else:
            event["resolved"] = False
            next_start = limit
        events.append(event)
    return events


def _calibration(events: list[dict], cutoff: int, side: str) -> dict:
    samples = [e for e in events if e["complete"] and e["resolved"]
               and e["phases"]["CHoCH"] + OUTCOME_WINDOW < cutoff]
    depths = [e["depth_atr"] for e in samples]
    moves = []
    for event in samples:
        future, entry, a = event["future"], event["entry"], event["atr"]
        move = (float(future.High.max()) - entry if side == "BUY"
                else entry - float(future.Low.min())) / a
        moves.append(max(0.0, move))
    sufficient = len(samples) >= MIN_SAMPLES
    return {"status": "CALIBRATED" if sufficient else "NOT_CALIBRATED",
            "sample_count": len(samples), "samples": samples,
            "depth_atr": float(np.quantile(depths, .95)) if sufficient else None,
            "moves_atr": moves if sufficient else [],
            "reason": "" if sufficient else
            f"فقط {len(samples)} نمونه کامل و غیرهمپوشان؛ حداقل {MIN_SAMPLES} لازم است. "
            "داده معتبر گروه نوسانی در دسترس نیست؛ جایگزین فرضی ساخته نشد."}


def _nearest_level(level_info: dict, side: str, entry: float) -> float | None:
    key = "nearest_major_resistance" if side == "BUY" else "nearest_major_support"
    level = _number(level_info.get(key))
    if level is None or not (level > entry if side == "BUY" else level < entry):
        return None
    return level


def _probability(samples: list[dict], side: str, reward_atr: float,
                 depth_atr: float) -> float:
    """برخورد همزمان حدسود/حدضرر در یک کندل، شکست محسوب می‌شود.

    نمونه بدون رسیدن به هدف در مهلت نیز موفقیت شمرده نمی‌شود.
    """
    wins = 0
    for event in samples:
        sign = 1 if side == "BUY" else -1
        stop = event["level"] - sign * depth_atr * event["atr"]
        target = event["entry"] + sign * reward_atr * event["atr"]
        if not (stop < event["entry"] if side == "BUY" else stop > event["entry"]):
            continue
        for candle in event["future"].itertuples():
            stopped = candle.Low <= stop if side == "BUY" else candle.High >= stop
            won = candle.High >= target if side == "BUY" else candle.Low <= target
            if stopped or won:
                wins += int(won and not stopped)
                break
    return round(100 * wins / len(samples), 2) if samples else 0.0


def _price_rows(side: str, entry: float, stop: float, atr: float,
                level: float, calibration: dict, min_rr: float) -> list[RiskRow]:
    if calibration["status"] != "CALIBRATED":
        return []
    sign = 1 if side == "BUY" else -1
    risk = sign * (entry - stop)
    room = sign * (level - entry)
    if risk <= 0 or room <= 0:
        return []
    rows = []
    # سه هدف از توزیع دامنه واقعی؛ برچسب‌های قدیمی برای سازگاری صفحه حفظ شده‌اند.
    for label, quantile in ((40, .25), (65, .5), (80, .75)):
        reward = min(room, float(np.quantile(calibration["moves_atr"], quantile)) * atr)
        rr = reward / risk
        if reward <= 0 or rr < min_rr:
            return []  # جدول ناقص سه‌ردیفی صادر نمی‌شود.
        probability = _probability(calibration["samples"], side, reward / atr,
                                   calibration["depth_atr"])
        rows.append(RiskRow(label, entry, stop, entry + sign * reward, rr,
                            probability, calibration["sample_count"]))
    return rows


def _confirmation(md: Any, timeframe: str, sensitivity: Any) -> dict:
    expected = dl.higher_timeframe(timeframe)
    answer = {"timeframe": expected, "side": None, "confirmed": False, "reason": ""}
    if expected is None:
        answer["reason"] = "تایم‌فریم بالاتر تعریف نشده است."
        return answer
    frame = getattr(md, "confirm_df", None)
    if getattr(md, "confirm_timeframe", None) != expected or not _valid_frame(frame):
        answer["reason"] = "داده معتبر تایم‌فریم تأیید موجود نیست."
        return answer
    # از داده آینده نسبت به آخرین کندل تایم‌فریم اصلی استفاده نمی‌شود.
    durations = {"15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D", "1W": "7D"}
    main_index = md.df.index
    main_index = main_index.tz_localize("UTC") if main_index.tz is None else main_index.tz_convert("UTC")
    main_end = min(main_index[-1] + pd.Timedelta(durations[timeframe]), pd.Timestamp.now(tz="UTC"))
    confirm_index = frame.index
    confirm_index = confirm_index.tz_localize("UTC") if confirm_index.tz is None else confirm_index.tz_convert("UTC")
    frame = frame.loc[confirm_index + pd.Timedelta(durations[expected]) <= main_end].tail(300)
    cfg = dl.TIMEFRAME_MATRIX[expected]
    if len(frame) < max(200, cfg["min_candles"]):
        answer["reason"] = "کمتر از ۲۰۰ کندل برای تأیید ساختاری وجود دارد."
        return answer
    highs, lows = ind.find_pivots(frame, _window(cfg, sensitivity))
    if len(highs) < 2 or len(lows) < 2:
        answer["reason"] = "پیوت کافی برای تأیید ساختار وجود ندارد."
        return answer
    up = frame.High.iloc[highs[-1]] > frame.High.iloc[highs[-2]]
    down = frame.Low.iloc[lows[-1]] < frame.Low.iloc[lows[-2]]
    higher_low = frame.Low.iloc[lows[-1]] > frame.Low.iloc[lows[-2]]
    lower_high = frame.High.iloc[highs[-1]] < frame.High.iloc[highs[-2]]
    answer["side"] = "BUY" if up and higher_low else "SELL" if down and lower_high else None
    answer["confirmed"] = answer["side"] is not None
    answer["reason"] = "" if answer["confirmed"] else "جهت ساختار تایم‌فریم بالاتر روشن نیست."
    return answer


def _empty_group(reason: str, **metadata: Any) -> dict:
    return {"entry": None, "rows": [], "action": "HOLD", "trigger": reason,
            "rr": 0.0, "hard_filters_passed": False,
            "hard_filter_reasons": [reason], **metadata}


def run_analysis(display_symbol: str, timeframe: str, side_preference: str,
                 rr_target: float = 1.5, pivot_sensitivity: Any = None,
                 require_htf_confirmation: bool = True) -> AnalysisResult:
    selected = ["BUY", "SELL"] if side_preference == "BOTH" else [side_preference]
    if any(side not in ("BUY", "SELL") for side in selected):
        return AnalysisResult(False, reason="جهت سیگنال معتبر نیست.")
    rr = _number(rr_target)
    if rr is None or rr < 1.5:
        return AnalysisResult(False, reason="حداقل نسبت سود به زیان باید ۱٫۵ یا بیشتر باشد.")
    if timeframe not in dl.TIMEFRAME_MATRIX:
        return AnalysisResult(False, reason="تایم‌فریم پشتیبانی نمی‌شود.")
    cfg = dl.TIMEFRAME_MATRIX[timeframe]
    try:
        window = _window(cfg, pivot_sensitivity)
    except ValueError as exc:
        return AnalysisResult(False, reason=str(exc))
    md = dl.load_market_data(display_symbol, timeframe)
    if not md.ok or not md.data_sufficient or not _valid_frame(md.df):
        return AnalysisResult(False, reason=md.reason or "داده OHLCV کافی و معتبر نیست.")
    df = md.df.tail(300).copy()
    if len(df) < 200:
        return AnalysisResult(False, reason="حداقل ۲۰۰ کندل واقعی برای بک‌تست لازم است.")
    # آخرین کندل باز حذف می‌شود؛ منبع داده موظف است زمان شروع کندل را ثبت کند.
    duration = {"15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D", "1W": "7D"}[timeframe]
    now = pd.Timestamp.now(tz="UTC")
    index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.loc[index + pd.Timedelta(duration) <= now]
    if len(df) < 200:
        return AnalysisResult(False, reason="پس از حذف کندل باز، کمتر از ۲۰۰ کندل باقی ماند.")
    atr_series = ind.compute_atr(df, cfg["atr_period"])
    a = _number(atr_series.iloc[-1]) if len(atr_series) else None
    if a is None or a <= 0:
        return AnalysisResult(False, reason="ATR معتبر موجود نیست؛ مقدار جایگزین ساخته نشد.")
    highs, lows = ind.find_pivots(df, window)
    if len(highs) < 2 or len(lows) < 2:
        return AnalysisResult(False, reason="پیوت کافی برای خطوط روند وجود ندارد.")
    upper = ind.build_trendline(df, highs, "RESISTANCE")
    lower = ind.build_trendline(df, lows, "SUPPORT")
    if upper is None or lower is None:
        return AnalysisResult(False, reason="ساخت خطوط روند معتبر ممکن نشد.")
    # تعداد برخورد به خط، درصد موفقیت تاریخی نیست.
    pattern = pat.analyze_pattern(df, upper, lower, None)
    level_info = lv.extract_levels(df, highs, lows)
    current = float(df.Close.iloc[-1])
    last_low, last_high = float(df.Low.iloc[lows[-1]]), float(df.High.iloc[highs[-1]])
    zones = {"BUY": liq.identify_liquidity_zone(df, last_low, a, "BUY_SIDE", "PIVOT", lower.strength_class),
             "SELL": liq.identify_liquidity_zone(df, last_high, a, "SELL_SIDE", "PIVOT", upper.strength_class)}
    # اشیای قدیمی صرفاً برای سازگاری نمایش باقی می‌مانند، نه مبنای قیمت‌گذاری.
    old_cal = {"BUY": liq.calibrate_stop_hunt(df, a, atr_series, lows, last_low, "BUY_SIDE", display_symbol),
               "SELL": liq.calibrate_stop_hunt(df, a, atr_series, highs, last_high, "SELL_SIDE", display_symbol)}
    confirmation = _confirmation(md, timeframe, pivot_sensitivity) if require_htf_confirmation else {
        "confirmed": False, "side": None, "timeframe": dl.higher_timeframe(timeframe),
        "reason": "تأیید تایم‌فریم بالاتر به درخواست کاربر الزامی نیست."}
    maturity = _number(getattr(pattern, "maturity_pct", None))
    status = getattr(pattern, "pattern_status", "UNKNOWN")
    built, phase_output, cal_output = {}, {}, {}
    for side in selected:
        events = _events(df, atr_series, side, max(2, min(window, 12)))
        latest = events[-1] if events else None
        cutoff = latest["sweep_index"] if latest else len(df)
        calibration = _calibration(events, cutoff, side)
        cal_output[side] = {k: v for k, v in calibration.items() if k not in ("samples", "moves_atr")}
        phases = latest["phases"] if latest else dict.fromkeys(PHASE_NAMES)
        phase_output[side] = {name: {"passed": phases[name] is not None,
                                    "candle_index": phases[name]} for name in PHASE_NAMES}
        reasons = []
        if latest is None or not latest["complete"]:
            missing = [name for name in PHASE_NAMES if phases[name] is None]
            reasons.append("مراحل کامل نیست: " + "، ".join(missing))
        elif len(df) - 1 - phases["CHoCH"] > EVENT_WINDOW:
            reasons.append("ستاپ اخیر نیست؛ از زمان تأیید ساختار بیش از ۱۲ کندل گذشته است.")
        if calibration["status"] != "CALIBRATED":
            reasons.append(calibration["reason"])
        if maturity is None or maturity > 85 or status in ("UNKNOWN", "FADING", "INVALID"):
            reasons.append("الگو نامعتبر است یا بلوغ آن از ۸۵٪ بیشتر است.")
        if require_htf_confirmation and confirmation["side"] != side:
            reasons.append(confirmation["reason"] or "جهت تایم‌فریم بالاتر با سیگنال همسو نیست.")
        groups = {}
        for name in ("standard", "stophunt", "liquidity_sweep"):
            metadata = {"calibration": old_cal[side], "liquidity_zone": zones[side],
                        "empirical_calibration": cal_output[side]} if name != "standard" else {}
            if reasons:
                groups[name] = _empty_group("؛ ".join(reasons), **metadata)
                continue
            sign = 1 if side == "BUY" else -1
            level = latest["level"]
            depth = calibration["depth_atr"] * a
            boundary = min(latest["extreme"], level - depth) if side == "BUY" else max(latest["extreme"], level + depth)
            # فاصله امن از صدک ۹۵ عمق تاریخی، بیرون از عمیق‌ترین شکار اخیر.
            buffer = max(depth * .05, np.spacing(boundary))
            stop = boundary - sign * buffer
            zone_low, zone_high = sorted((boundary, level))
            if name == "standard":
                entry = current
            elif name == "stophunt":
                entry = level  # بازآزمایی سطح بازپس‌گرفته‌شده، نه زیر مرز امن.
            else:
                entry = (zone_low + zone_high) / 2
            group_reasons = []
            after_confirmation = df.iloc[latest["phases"]["CHoCH"] + 1:]
            if not after_confirmation.empty and (
                float(after_confirmation.Low.min()) <= stop if side == "BUY"
                else float(after_confirmation.High.max()) >= stop
            ):
                group_reasons.append("پس از تأیید ساختار حدضرر نقض شده؛ ستاپ باطل است.")
            if name != "standard" and not (entry < current if side == "BUY" else entry > current):
                group_reasons.append("قیمت Limit در سمت معتبر قیمت فعلی قرار ندارد.")
            if not (stop < zone_low if side == "BUY" else stop > zone_high):
                group_reasons.append("حدضرر بیرون ناحیه شکار نیست.")
            target_level = _nearest_level(level_info, side, entry)
            if target_level is None:
                group_reasons.append("سطح قوی واقعی در جهت هدف موجود نیست.")
            rows = [] if group_reasons else _price_rows(side, entry, stop, a, target_level, calibration, rr)
            if not rows:
                group_reasons.append("سه هدف تاریخی با حداقل نسبت سود به زیان قابل تأیید نیست.")
            # نمونه‌ها با ورود CHoCH اندازه‌گیری شدند؛ احتمال آن‌ها برای Limit قابل تعمیم نیست.
            if name != "standard":
                group_reasons.append("بک‌تست اختصاصی ورود Limit در این نسخه از فایل نقدینگی موجود نیست؛ احتمال آن تخمین زده نشد.")
            if group_reasons:
                groups[name] = _empty_group("؛ ".join(group_reasons), **metadata,
                                           sweep_zone={"lower": zone_low, "upper": zone_high})
                continue
            groups[name] = {"entry": entry, "rows": rows, "action": side,
                            "trigger": "چهار مرحله، کالیبراسیون و فیلترها تأیید شدند.",
                            "rr": min(r.rr for r in rows), "hard_filters_passed": True,
                            "hard_filter_reasons": [], **metadata}
        std = groups["standard"]
        # امتیاز نمایشی قدیمی برای تصمیم یا احتمال موفقیت مصرف نمی‌شود.
        built[side] = {"side": side, **groups, "ladder": {}, "confidence": None,
                       "hard_filters_passed": std["hard_filters_passed"],
                       "hard_filter_reasons": std["hard_filter_reasons"],
                       "final_decision": std["action"]}
    actionable = [s for s in selected if built[s]["final_decision"] != "HOLD"]
    if len(actionable) > 1:
        for side in selected:
            for name in ("standard", "stophunt", "liquidity_sweep"):
                built[side][name] = _empty_group("تعارض جهت خرید و فروش؛ هیچ سفارش صادر نشد.")
            built[side].update(final_decision="HOLD", hard_filters_passed=False,
                               hard_filter_reasons=["تعارض جهت خرید و فروش."])
        actionable = []
    primary = built[actionable[0] if actionable else selected[0]]
    reason = "" if actionable else "؛ ".join(dict.fromkeys(
        reason for side in selected for reason in built[side]["hard_filter_reasons"]))
    return AnalysisResult(
        ok=True, reason=reason,
        market={"symbol": display_symbol, "timeframe": timeframe, "current_price": current,
                "candles_analyzed": len(df), "atr_value": a, "atr_period": cfg["atr_period"],
                "pivot_window": window, "data_source": md.data_source,
                "data_sufficient": True, "reason": md.reason, "df": df},
        upper_line=upper, lower_line=lower, pattern=pattern, level_info=level_info,
        liquidity_zone_buy=zones["BUY"], liquidity_zone_sell=zones["SELL"],
        stophunt_buy=old_cal["BUY"], stophunt_sell=old_cal["SELL"],
        standard_signal={s: built[s]["standard"] for s in selected},
        stophunt_signal={s: built[s]["stophunt"] for s in selected},
        liquidity_sweep_signal={s: built[s]["liquidity_sweep"] for s in selected},
        hard_filters_passed=primary["hard_filters_passed"],
        hard_filter_reasons=primary["hard_filter_reasons"], final_decision=primary["final_decision"],
        sides=built, phases=phase_output, calibration=cal_output, confirmation=confirmation)