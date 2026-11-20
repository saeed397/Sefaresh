"""core/signals.py — سیگنال‌های ایمن بر پایه شواهد تاریخی.

نام توابع و فیلدهای قدیمی حفظ شده‌اند؛ ورودی‌های جدید اختیاری‌اند.
نبود چهار مرحله یا بک‌تست اختصاصی یعنی HOLD و جدول خالی.
برچسب‌های ۴۰/۶۵/۸۰ برای سازگاری‌اند، نه درصد سرمایه یا احتمال موفقیت.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

MIN_RR = 1.5
MIN_SAMPLES = 5
PHASE_NAMES = ('Sweep', 'Reclaim', 'Displacement', 'CHoCH')


@dataclass
class RiskRow:
    risk_pct: int
    take_profit: float
    entry: float
    stop_loss: float
    rr: float
    success_probability: float | None = None
    sample_count: int = 0
    risk_label: str = 'برچسب سناریو؛ احتمال فقط از نتیجه تاریخی محاسبه می‌شود'


@dataclass
class SignalGroup:
    group_name: str
    side: str
    entry_price: float | None
    rows: list
    trigger: str


@dataclass
class ConfidenceScore:
    total: float
    score_class: str
    components: dict


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _positive(value):
    number = _number(value)
    return number is not None and number > 0


def _ordered(phases):
    if not isinstance(phases, dict):
        return False
    indices = [phases.get(name) for name in PHASE_NAMES]
    return (all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
                and i >= 0 for i in indices)
            and all(a < b for a, b in zip(indices, indices[1:])))


def _evidence_reasons(side, phases, calibration_status, htf_confirmed,
                      require_htf_confirmation, historical_candles):
    reasons = []
    if side not in ('BUY', 'SELL'):
        reasons.append('جهت معتبر نیست')
    if not _ordered(phases):
        reasons.append('چهار مرحله Sweep، Reclaim، Displacement، CHoCH کامل و مرتب نیست')
    if calibration_status != 'CALIBRATED':
        reasons.append('کالیبراسیون تاریخی اختصاصی موجود نیست')
    count = _number(historical_candles)
    if count is None or count != int(count) or not 200 <= count <= 300:
        reasons.append('بک‌تست به ۲۰۰ تا ۳۰۰ کندل تاریخی نیاز دارد')
    if require_htf_confirmation and htf_confirmed is not True:
        reasons.append('تأیید هم‌جهت تایم‌فریم بالاتر موجود نیست')
    return reasons


def build_risk_ladder(side: str, entry: float, base_stop: float, atr: float,
                      rr_target: float, risk_levels=(40, 65, 80), *,
                      historical_samples=None, opposing_level=None,
                      sweep_boundary=None, phases=None,
                      calibration_status='NOT_CALIBRATED', historical_candles=None,
                      htf_confirmed=False, require_htf_confirmation=True) -> list[RiskRow]:
    """سه هدف از دامنه تاریخی؛ حدضرر مشترک بیرون مرز شکار، بدون گسترش فرضی.

    هر نمونه باید کامل، resolved و غیرهمپوشان باشد و entry، atr، level،
    phases و future (جدول High/Low) داشته باشد. آخرین نتیجه باید پیش از
    Sweep فعلی تمام شود. نمونه‌ها متعلق به همان نماد/تایم‌فریم ورودی‌اند.
    ورود این نمونه‌ها پس از CHoCH است؛ برای Limit قابل تعمیم نیست.
    """
    if _evidence_reasons(side, phases, calibration_status, htf_confirmed,
                         require_htf_confirmation, historical_candles):
        return []
    if tuple(risk_levels) != (40, 65, 80):
        return []
    values = [entry, base_stop, atr, rr_target, opposing_level, sweep_boundary]
    if not all(_positive(v) for v in values) or rr_target < MIN_RR:
        return []
    sign = 1 if side == 'BUY' else -1
    risk, room = sign * (entry - base_stop), sign * (opposing_level - entry)
    if risk <= 0 or room <= 0 or sign * (sweep_boundary - base_stop) <= 0:
        return []
    samples, last_end = [], -1
    for sample in sorted(historical_samples or [],
                         key=lambda s: s.get('phases', {}).get('Sweep', -1)):
        if not sample.get('complete') or not sample.get('resolved'):
            continue
        old = sample.get('phases')
        if not _ordered(old) or old['Sweep'] <= last_end:
            continue
        future = sample.get('future')
        if future is None or not {'High', 'Low'}.issubset(future.columns) or len(future) != 12:
            continue
        end = old['CHoCH'] + len(future)
        if end >= phases['Sweep']:
            continue
        if not all(_positive(sample.get(k)) for k in ('entry', 'atr', 'level')):
            continue
        values = future[['High', 'Low']].to_numpy(float)
        if not (np.isfinite(values).all() and (values > 0).all()
                and (values[:, 0] >= values[:, 1]).all()):
            continue
        # همان هندسه حدضرر در گذشته؛ نمونه نامعتبر حذف نمی‌شود تا نرخ خوش‌بینانه نشود.
        samples.append(sample)
        last_end = end
    if len(samples) < MIN_SAMPLES:
        return []
    moves = [max(0., (float(s['future'].High.max()) - s['entry']) / s['atr']
                 if side == 'BUY' else (s['entry'] - float(s['future'].Low.min())) / s['atr'])
             for s in samples]
    rows = []
    for label, q in zip(risk_levels, (.25, .5, .75)):
        reward = min(room, float(np.quantile(moves, q)) * atr)
        rr = reward / risk
        if reward <= 0 or rr < rr_target:
            return []
        wins = 0
        for sample in samples:
            stop = sample['entry'] - sign * (risk / atr) * sample['atr']
            target = sample['entry'] + sign * reward / atr * sample['atr']
            for candle in sample['future'].itertuples():
                stopped = candle.Low <= stop if side == 'BUY' else candle.High >= stop
                won = candle.High >= target if side == 'BUY' else candle.Low <= target
                if stopped or won:
                    wins += int(won and not stopped)
                    break
        rows.append(RiskRow(label, entry + sign * reward, entry, base_stop, rr,
                            round(100. * wins / len(samples), 2), len(samples)))
    return rows


def structural_stop_loss(side: str, last_pivot_price: float, atr: float, *,
                         historical_buffer=None) -> float | None:
    """بدون بافر اندازه‌گیری‌شده قیمت فرضی ساخته نمی‌شود."""
    if side not in ('BUY', 'SELL') or not all(_positive(v) for v in
            (last_pivot_price, atr, historical_buffer)):
        return None
    result = last_pivot_price + (-1 if side == 'BUY' else 1) * historical_buffer
    return float(result) if result > 0 else None


def anti_hunt_stop_loss(side: str, liquidity_boundary: float, atr: float, *,
                        historical_buffer=None) -> float | None:
    return structural_stop_loss(side, liquidity_boundary, atr,
                                historical_buffer=historical_buffer)


def immediate_signal(side: str, current_price: float, upper_strength: str,
                     lower_strength: str, breakout_vol_ratio: float,
                     pattern_status: str, maturity_pct: float,
                     rr_check: float, *, phases=None,
                     calibration_status='NOT_CALIBRATED', historical_candles=None,
                     htf_confirmed=False, require_htf_confirmation=True) -> tuple[str, str]:
    reasons = _evidence_reasons(side, phases, calibration_status, htf_confirmed,
                                require_htf_confirmation, historical_candles)
    if not _positive(current_price) or _number(rr_check) is None or rr_check < MIN_RR:
        reasons.append('قیمت یا نسبت سود به زیان معتبر نیست؛ حداقل نسبت ۱٫۵ است')
    if _number(maturity_pct) is None or not 0 <= maturity_pct <= 85:
        reasons.append('بلوغ الگو نامعلوم یا بیش از ۸۵٪ است')
    strength = lower_strength if side == 'BUY' else upper_strength
    if strength not in ('MEDIUM', 'STRONG'):
        reasons.append('خط ساختاری معتبر و کافی نیست')
    if pattern_status not in ('BREAKING_OUT', 'APPROACHING_BREAKOUT', 'CONSOLIDATING', 'VALID'):
        reasons.append('الگو معتبر نیست')
    if pattern_status == 'BREAKING_OUT' and (
            _number(breakout_vol_ratio) is None or breakout_vol_ratio < 1.2):
        reasons.append('تأیید حجم شکست کافی نیست')
    if reasons:
        return 'HOLD', '؛ '.join(reasons)
    return ('LONG' if side == 'BUY' else 'SHORT'), 'چهار مرحله و شواهد تاریخی تأیید شدند.'


def limit_signal(side: str, hypothesis_status: str, calibration_status: str,
                 rr_check: float, **evidence) -> tuple[str, str]:
    """بک‌تست اختصاصی اجرای Limit موجود نیست؛ سفارش صادر نمی‌شود."""
    return 'HOLD', 'بک‌تست اختصاصی ورود Limit تأیید نشده؛ فرضیه شکار به‌تنهایی مجوز سفارش نیست.'


def compute_confidence(upper_score: float, lower_score: float, breakout_vol_ratio: float,
                       rr: float, maturity_pct: float, sr_confluence: bool,
                       stophunt_quality: float) -> ConfidenceScore:
    """امتیاز ساختاری نمایشی، نه احتمال برد یا مجوز عبور از فیلترها."""
    def bounded(value, limit, weight):
        n = _number(value)
        return 0. if n is None else float(np.clip(n / limit, 0, 1)) * weight
    volume, ratio, maturity = map(_number, (breakout_vol_ratio, rr, maturity_pct))
    parts = {
        'upper_line_strength': bounded(upper_score, 100., 20.),
        'lower_line_strength': bounded(lower_score, 100., 20.),
        'breakout_volume': bounded(None if volume is None else volume - 1., 1., 15.),
        'risk_reward': bounded(None if ratio is None else ratio - MIN_RR, 1.5, 15.),
        'pattern_maturity': (10. if maturity is not None and 50 <= maturity <= 75
                             else 6. if maturity is not None and 40 <= maturity <= 85 else 0.),
        'sr_confluence': 10. if sr_confluence is True else 0.,
        'stophunt_quality': bounded(stophunt_quality, 100., 10.),
    }
    total = round(sum(parts.values()), 1)
    cls = next((name for threshold, name in ((85, 'VERY_STRONG'), (70, 'VALID'),
                (55, 'MEDIUM'), (40, 'WEAK')) if total >= threshold), 'REJECTED')
    return ConfidenceScore(total, cls, {k: round(v, 1) for k, v in parts.items()})


def apply_hard_filters(data_sufficient: bool, rr: float, entry_defined: bool,
                       stop_defined: bool, sl_inside_liquidity: bool,
                       pattern_valid: bool, *, side=None, phases=None,
                       calibration_status='NOT_CALIBRATED', historical_candles=None,
                       htf_confirmed=False, require_htf_confirmation=True,
                       maturity_pct=None, target_within_level=False,
                       entry_safe=False) -> tuple[bool, list[str]]:
    reasons = _evidence_reasons(side, phases, calibration_status, htf_confirmed,
                                require_htf_confirmation, historical_candles)
    for failed, reason in (
        (data_sufficient is not True, 'کمبود داده تاریخی'),
        (_number(rr) is None or rr < MIN_RR, 'R:R کمتر از ۱٫۵ یا نامعلوم است'),
        (entry_defined is not True or stop_defined is not True, 'ورود یا حدضرر مشخص نیست'),
        (sl_inside_liquidity is not False, 'حدضرر بیرون ناحیه نقدینگی تأیید نشده'),
        (pattern_valid is not True, 'الگو معتبر نیست'),
        (_number(maturity_pct) is None or not 0 <= maturity_pct <= 85, 'بلوغ الگو معتبر نیست'),
        (target_within_level is not True, 'هدف با سطح قوی واقعی محدود نشده'),
        (entry_safe is not True, 'ورود ایمن تأیید نشده'),
    ):
        if failed:
            reasons.append(reason)
    return not reasons, reasons


def final_decision(action: str, hard_filters_passed: bool, score_class: str) -> str:
    if hard_filters_passed is not True or score_class not in ('VERY_STRONG', 'VALID', 'MEDIUM', 'WEAK'):
        return 'HOLD'
    # Limit تا بک‌تست اختصاصی مسدود است؛ اقدام خرید/فروش با HOLD همزمان صادر نمی‌شود.
    return {'LONG': 'EXECUTE_LONG', 'SHORT': 'EXECUTE_SHORT',
            'BUY': 'BUY', 'SELL': 'SELL'}.get(action, 'HOLD')
