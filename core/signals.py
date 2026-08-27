"""
core/signals.py
----------------
فصل ۹: سیگنال فوری
فصل ۱۳: مدیریت حدضرر و ریسک
فصل ۱۴: امتیاز اطمینان ساختاری (غیرذهنی)
فصل ۱۶: منطق تصمیم‌گیری نهایی و فیلترهای سخت
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_RR = 1.5


@dataclass
class RiskRow:
    risk_pct: int
    take_profit: float
    entry: float
    stop_loss: float
    rr: float


@dataclass
class SignalGroup:
    group_name: str  # "STANDARD" | "STOPHUNT"
    side: str  # BUY | SELL
    entry_price: float
    rows: list  # list[RiskRow]
    trigger: str


@dataclass
class ConfidenceScore:
    total: float
    score_class: str
    components: dict


def build_risk_ladder(side: str, entry: float, base_stop: float, atr: float,
                       rr_target: float, risk_levels=(40, 65, 80)) -> list[RiskRow]:
    """
    توضیح ۱ سند: سه ردیف دقیق با ریسک ۴۰٪، ۶۵٪ و ۸۰٪.
    درصد ریسک بالاتر یعنی حدضرر دورتر (تحمل نوسان بیشتر) و در نتیجه حدسود دورتر
    (متناسب با R:R انتخابی کاربر) — یعنی سناریوی تهاجمی‌تر با هدف بزرگ‌تر.
    """
    direction = 1 if side == "BUY" else -1
    rows = []
    base_risk_dist = abs(entry - base_stop)
    if base_risk_dist <= 0:
        base_risk_dist = atr * 0.5

    for pct in risk_levels:
        risk_mult = 1.0 + (pct / 100.0) * 1.2  # مقیاس‌دهی فاصله حدضرر بر مبنای درصد ریسک
        stop_dist = base_risk_dist * risk_mult
        stop_loss = entry - direction * stop_dist
        reward_dist = stop_dist * rr_target
        take_profit = entry + direction * reward_dist
        rr = reward_dist / stop_dist if stop_dist else 0.0
        rows.append(RiskRow(risk_pct=pct, take_profit=float(take_profit),
                             entry=float(entry), stop_loss=float(stop_loss), rr=float(rr)))
    return rows


def structural_stop_loss(side: str, last_pivot_price: float, atr: float) -> float:
    """فصل ۱۳.۱: حدضرر ساختاری — آخرین Pivot معتبر + بافر کوچک."""
    buffer = atr * 0.15
    return last_pivot_price - buffer if side == "BUY" else last_pivot_price + buffer


def anti_hunt_stop_loss(side: str, liquidity_boundary: float, atr: float) -> float:
    """فصل ۱۳.۲: حدضرر ضد شکار — خارج از مرز نهایی باکس نقدینگی."""
    buffer = atr * 0.25
    return liquidity_boundary - buffer if side == "BUY" else liquidity_boundary + buffer


def immediate_signal(side: str, current_price: float, upper_strength: str,
                      lower_strength: str, breakout_vol_ratio: float,
                      pattern_status: str, maturity_pct: float,
                      rr_check: float) -> tuple[str, str]:
    """فصل ۹: سناریوی LONG/SHORT یا HOLD/CANCEL_LOW_RR."""
    weak_line = (side == "BUY" and lower_strength == "WEAK") or (side == "SELL" and upper_strength == "WEAK")
    low_volume = breakout_vol_ratio < 1.2
    over_mature = maturity_pct > 85
    low_rr = rr_check < MIN_RR

    if low_rr:
        return "CANCEL_LOW_RR", "نسبت ریسک به ریوارد کمتر از حداقل مجاز (۱:۱.۵) است."
    if weak_line and low_volume:
        return "HOLD", "قدرت خط ضعیف و حجم شکست ناکافی — ورود فوری توصیه نمی‌شود."
    if over_mature:
        return "HOLD", "بلوغ الگو بیش از ۸۵٪ — احتمال Fading بالاست."
    if pattern_status == "BREAKING_OUT" and not low_volume:
        action = "LONG" if side == "BUY" else "SHORT"
        return action, "شکست معتبر ساختار همراه با تأیید حجمی."
    if pattern_status in ("APPROACHING_BREAKOUT", "CONSOLIDATING"):
        action = "LONG" if side == "BUY" else "SHORT"
        return action, "واکنش به دیوار ساختاری با شرایط قابل قبول."
    return "HOLD", "شرایط تأیید کامل شکست/واکنش هنوز فراهم نیست."


def limit_signal(side: str, hypothesis_status: str, calibration_status: str,
                  rr_check: float) -> tuple[str, str]:
    """سیگنال دوم بر مبنای فرضیه Stop-Hunt."""
    if rr_check < MIN_RR:
        return "CANCEL_LOW_RR", "نسبت ریسک به ریوارد سیگنال لیمیت کمتر از حد مجاز است."
    action = "LIMIT_LONG" if side == "BUY" else "LIMIT_SHORT"
    trigger = (
        "سفارش معلق روی ناحیه احتمالی تجمع نقدینگی؛ فعال‌سازی صرفاً با رسیدن قیمت "
        "به محدوده و شکار نقدینگی (Sweep) رخ می‌دهد."
    )
    if calibration_status == "NOT_CALIBRATED":
        trigger += " توجه: کالیبراسیون اختصاصی این دارایی کامل نیست؛ از بازه عمومی مرجع استفاده شد."
    return action, trigger


def compute_confidence(upper_score: float, lower_score: float, breakout_vol_ratio: float,
                        rr: float, maturity_pct: float, sr_confluence: bool,
                        stophunt_quality: float) -> ConfidenceScore:
    """
    فصل ۱۴.۲: ساختار امتیاز کل از ۱۰۰
    قدرت خط بالایی ۲۰ / قدرت خط پایینی ۲۰ / تأیید حجم شکست ۱۵ /
    R:R ۱۵ / بلوغ مناسب الگو ۱۰ / همگرایی حمایت-مقاومت ۱۰ / کیفیت Stop-Hunt ۱۰
    """
    c_upper = (upper_score / 100.0) * 20
    c_lower = (lower_score / 100.0) * 20

    vol_norm = np.clip((breakout_vol_ratio - 1.0) / 1.0, 0, 1)  # ۱x تا ۲x -> ۰ تا ۱
    c_vol = vol_norm * 15

    rr_norm = np.clip((rr - MIN_RR) / (3.0 - MIN_RR), 0, 1)
    c_rr = rr_norm * 15

    # بلوغ مطلوب ۵۰-۷۵٪ (فصل ۷.۱)
    if 50 <= maturity_pct <= 75:
        c_mat = 10.0
    elif 40 <= maturity_pct < 50 or 75 < maturity_pct <= 85:
        c_mat = 6.0
    else:
        c_mat = 2.0

    c_confluence = 10.0 if sr_confluence else 4.0
    c_stophunt = (stophunt_quality / 100.0) * 10

    total = c_upper + c_lower + c_vol + c_rr + c_mat + c_confluence + c_stophunt
    total = float(np.clip(total, 0, 100))

    if total >= 85:
        cls = "VERY_STRONG"
    elif total >= 70:
        cls = "VALID"
    elif total >= 55:
        cls = "MEDIUM"
    elif total >= 40:
        cls = "WEAK"
    else:
        cls = "REJECTED"

    return ConfidenceScore(
        total=round(total, 1), score_class=cls,
        components={
            "upper_line_strength": round(c_upper, 1),
            "lower_line_strength": round(c_lower, 1),
            "breakout_volume": round(c_vol, 1),
            "risk_reward": round(c_rr, 1),
            "pattern_maturity": round(c_mat, 1),
            "sr_confluence": round(c_confluence, 1),
            "stophunt_quality": round(c_stophunt, 1),
        },
    )


def apply_hard_filters(data_sufficient: bool, rr: float, entry_defined: bool,
                        stop_defined: bool, sl_inside_liquidity: bool,
                        pattern_valid: bool) -> tuple[bool, list[str]]:
    """فصل ۱۶: فیلترهای سخت — حتی با امتیاز بالا رد می‌شود."""
    reasons = []
    if not data_sufficient:
        reasons.append("کمبود داده تاریخی")
    if rr < MIN_RR:
        reasons.append("R:R کمتر از ۱:۱.۵")
    if not entry_defined or not stop_defined:
        reasons.append("ورود یا حدضرر مشخص نیست")
    if sl_inside_liquidity:
        reasons.append("حدضرر داخل ناحیه نقدینگی قرار دارد")
    if not pattern_valid:
        reasons.append("الگوی شناسایی‌شده معتبر نیست")
    return (len(reasons) == 0), reasons


def final_decision(action: str, hard_filters_passed: bool, score_class: str) -> str:
    if not hard_filters_passed:
        return "CANCEL_LOW_RR" if action not in ("HOLD",) else "HOLD"
    if action in ("HOLD", "CANCEL_LOW_RR"):
        return action
    if score_class == "REJECTED":
        return "HOLD"
    mapping = {
        "LONG": "EXECUTE_LONG", "SHORT": "EXECUTE_SHORT",
        "LIMIT_LONG": "PLACE_LIMIT_LONG", "LIMIT_SHORT": "PLACE_LIMIT_SHORT",
    }
    return mapping.get(action, "HOLD")
