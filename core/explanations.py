"""
explanations.py — متن‌های ساده دکمه‌های ℹ️

قاعده: هیچ عدد یا وعده ثابتی در متن‌ها نوشته نمی‌شود. توضیح هر بخش از روی
اعداد واقعی همان تحلیل (خروجی engine.run_analysis) ساخته می‌شود. اگر عددی
وجود ندارد، صادقانه گفته می‌شود «معلوم نیست» و چیزی حدس زده نمی‌شود.
"""
from __future__ import annotations

from typing import Any

# توضیح پایه هر دکمه (بدون عدد). کلیدها همان کلیدهای قبلی هستند.
EXPLANATIONS: dict[str, str] = {
    "line_strength": "قدرت خط یعنی قیمت چند بار به این خط خورده و برگشته. "
                     "این تعداد، درصد موفقیت نیست؛ فقط نشان می‌دهد خط چقدر دیده شده.",
    "pattern": "الگو شکل کلی حرکت قیمت بین دو خط است. اگر الگو زیادی جلو رفته باشد "
               "(بیش از ۸۵٪)، دیر است و سیگنال داده نمی‌شود.",
    "standard_box": "ورود معمولی: با قیمت فعلی بازار وارد می‌شوید، فقط وقتی هر چهار "
                    "مرحله کامل شده باشد.",
    "stophunt_box": "ورود Limit بعد از شکار حدضرر: سفارش روی سطحی گذاشته می‌شود که "
                    "قیمت دوباره پس گرفته. تا بک‌تست مخصوص آن آماده نشود، قیمتی داده نمی‌شود.",
    "liquidity_sweep": "ناحیه جاروی نقدینگی: محدوده‌ای که قیمت برای گرفتن حدضررها "
                       "به آن سر زده و برگشته است.",
    "support_resistance": "حمایت کفی است که قیمت معمولاً از آن بالا می‌رود؛ مقاومت "
                          "سقفی است که قیمت معمولاً از آن پایین می‌آید. حدسود از این سقف/کف رد نمی‌شود.",
    "confidence_score": "اطمینان از روی نمونه‌های واقعی گذشته همین نماد و تایم‌فریم حساب "
                        "می‌شود. اگر نمونه کافی نباشد، عددی ساخته نمی‌شود.",
    "risk_percent": "۴۰، ۶۵ و ۸۰ فقط اسم سه ردیف هدف (نزدیک، میانه، دور) است؛ "
                    "درصد سرمایه شما نیست.",
    "data_source": "قیمت‌ها از Yahoo Finance، در صورت قطعی از Kraken و بعد CoinGecko "
                   "گرفته می‌شود. از بایننس و داده ساختگی استفاده نمی‌شود.",
    "rr": "نسبت سود به زیان یعنی در برابر هر یک واحد ریسک، چند واحد سود ممکن است. "
          "حداقل قابل قبول ۱٫۵ است.",
    "phases": "چهار مرحله به ترتیب: جارو (Sweep)، پس‌گرفتن سطح (Reclaim)، حرکت قوی "
              "(Displacement)، تغییر ساختار (CHoCH). بدون هر چهار مرحله، سیگنالی نیست.",
    "confirmation": "تایم‌فریم بالاتر باید هم‌جهت باشد: 15m←1h، 1h←4h، 4h←1D، 1D←1W.",
}

_SIDE_FA = {"BUY": "خرید", "SELL": "فروش", "HOLD": "صبر (بدون معامله)"}


def get_explanation(key: str) -> str:
    """سازگار با نسخه قبلی: متن پایه یک دکمه."""
    return EXPLANATIONS.get(key, "توضیحی برای این بخش موجود نیست.")


def _num(value: Any, digits: int = 2) -> str | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else f"{v:,.{digits}f}"


def _group_text(group: dict | None, title: str) -> str:
    if not group:
        return f"{title}: اطلاعاتی ساخته نشد."
    if group.get("action") == "HOLD" or not group.get("rows"):
        reason = group.get("trigger") or "شرایط کامل نیست."
        return f"{title}: فعلاً صبر کنید. دلیل: {reason}"
    lines = [f"{title}: {_SIDE_FA.get(group['action'], group['action'])} از قیمت "
             f"{_num(group.get('entry'))}."]
    for row in group["rows"]:
        prob = _num(getattr(row, "success_probability", None) * 100
                    if getattr(row, "success_probability", None) is not None else None, 0)
        lines.append(
            f"• هدف {row.risk_percent}: حدسود {_num(row.take_profit)}، حدضرر "
            f"{_num(row.stop_loss)}، سود به زیان {_num(row.rr)}، "
            f"موفقیت در گذشته {prob + '٪' if prob else 'نامعلوم'} "
            f"(از {row.sample_count} نمونه واقعی).")
    return "\n".join(lines)


def build_explanations(result: Any, side: str) -> dict[str, str]:
    """ماتریس شرطی: توضیح هر دکمه از روی اعداد واقعی همین تحلیل."""
    out = dict(EXPLANATIONS)
    if result is None or not getattr(result, "ok", False):
        reason = getattr(result, "reason", "") or "تحلیل انجام نشد."
        return {k: f"{v}\n\nوضعیت فعلی: تحلیل متوقف شد — {reason}" for k, v in out.items()}

    sides = getattr(result, "sides", {}) or {}
    data = sides.get(side, {})
    market = getattr(result, "market", {}) or {}

    src = market.get("data_source")
    count = market.get("candles_analyzed")
    out["data_source"] += f"\n\nاین بار: منبع {src or 'نامعلوم'}، {count or 'نامعلوم'} کندل بسته‌شده."

    pattern = getattr(result, "pattern", None)
    maturity = _num(getattr(pattern, "maturity_pct", None), 0)
    status = getattr(pattern, "pattern_status", "UNKNOWN")
    if maturity is None:
        out["pattern"] += "\n\nاین بار: بلوغ الگو معلوم نیست، پس سیگنالی صادر نمی‌شود."
    elif float(maturity.replace(",", "")) > 85:
        out["pattern"] += f"\n\nاین بار: بلوغ {maturity}٪ — دیر است، صبر کنید."
    else:
        out["pattern"] += f"\n\nاین بار: بلوغ {maturity}٪، وضعیت {status}."

    for key, line in (("line_strength", "lower_line" if side == "BUY" else "upper_line"),):
        cls = getattr(getattr(result, line, None), "strength_class", None)
        out[key] += f"\n\nاین بار: قدرت خط {'نامعلوم' if cls is None else cls}."

    lv = getattr(result, "level_info", {}) or {}
    res, sup = _num(lv.get("nearest_major_resistance")), _num(lv.get("nearest_major_support"))
    out["support_resistance"] += f"\n\nاین بار: مقاومت {res or 'پیدا نشد'}، حمایت {sup or 'پیدا نشد'}."

    phases = (getattr(result, "phases", {}) or {}).get(side, {})
    if phases:
        done = [n for n, p in phases.items() if p.get("passed")]
        miss = [n for n, p in phases.items() if not p.get("passed")]
        out["phases"] += f"\n\nاین بار: کامل‌شده {('، '.join(done)) or 'هیچ‌کدام'}؛ " \
                         f"ناقص {('، '.join(miss)) or 'هیچ‌کدام'}."

    conf = getattr(result, "confirmation", {}) or {}
    tf = conf.get("timeframe") or "بالاتر"
    agree = conf.get("side") == side
    out["confirmation"] += (f"\n\nاین بار: تایم‌فریم {tf} "
                            f"{'هم‌جهت است.' if agree else 'هم‌جهت نیست — ' + (conf.get('reason') or '')}")

    cal = (getattr(result, "calibration", {}) or {}).get(side, {})
    n = cal.get("sample_count", 0)
    out["confidence_score"] += (f"\n\nاین بار: {n} نمونه واقعی؛ "
                                + ("کافی است." if cal.get("status") == "CALIBRATED"
                                   else "کافی نیست، پس احتمالی گفته نمی‌شود."))

    std, sh, ls = data.get("standard"), data.get("stophunt"), data.get("liquidity_sweep")
    out["standard_box"] += "\n\n" + _group_text(std, "ورود معمولی")
    out["stophunt_box"] += "\n\n" + _group_text(sh, "ورود Limit")
    zone = (ls or {}).get("sweep_zone")
    out["liquidity_sweep"] += "\n\n" + (
        f"ناحیه: از {_num(zone['lower'])} تا {_num(zone['upper'])}." if zone else "ناحیه‌ای ثبت نشد.")
    rr = _num((std or {}).get("rr")) if std and std.get("rows") else None
    out["rr"] += f"\n\nاین بار: {'کمترین نسبت ' + rr if rr else 'نسبتی تأیید نشد.'}"
    decision = getattr(result, "final_decision", "HOLD")
    out["final_decision"] = f"تصمیم نهایی: {_SIDE_FA.get(decision, decision)}."
    return out