"""توضیح کاربردی هر بخش از اطلاعات همان تحلیل؛ بدون قیمت یا احتمال ساختگی.
رابط EXPLANATIONS، get_explanation و build_explanations حفظ شده است.
صفحه باید build_explanations(result, side) را برای هر تحلیل فراخوانی کند.
"""
from __future__ import annotations
from math import isfinite
from typing import Any

EXPLANATIONS = {
    'line_strength': 'قیمت هرچه بیشتر به یک خط واکنش نشان داده باشد، آن خط بیشتر مورد توجه بوده؛ این امتیاز تضمین برگشت قیمت نیست.',
    'pattern': 'شکل حرکت قیمت کمک می‌کند وضعیت بازار را بشناسیم؛ اسم الگو به‌تنهایی اجازه ورود نمی‌دهد.',
    'standard_box': 'این گروه ورود با قیمت فعلی را بررسی می‌کند؛ مجوز ورود به نتیجه فیلترها بستگی دارد.',
    'stophunt_box': 'این گروه برگشت قیمت به محل تجمع حدضررها را بررسی می‌کند؛ لمس محدوده به‌تنهایی تأیید برگشت نیست.',
    'liquidity_sweep': 'جاروی نقدینگی یعنی قیمت از یک سطح حساس عبور کند و دوباره آن را پس بگیرد.',
    'support_resistance': 'حمایت محل واکنش زیر قیمت و مقاومت محل واکنش بالای قیمت است؛ سطح قوی به برخوردهای مستقل بیشتری نیاز دارد.',
    'confidence_score': 'امتیاز ساختاری و موفقیت تاریخی دو چیز متفاوت‌اند؛ هیچ‌کدام تضمین سود نیستند.',
    'risk_percent': '۴۰، ۶۵ و ۸۰ درجه تهاجمی‌بودن سه سناریو هستند، نه احتمال برد و نه درصد سرمایه‌ای که باید وارد معامله کنید.',
    'data_source': 'داده از Yahoo Finance، سپس Kraken و بعد CoinGecko گرفته می‌شود؛ داده بایننس استفاده نمی‌شود.',
    'rr': 'نسبت سود به زیان از فاصله واقعی ورود تا هدف و حدضرر محاسبه می‌شود، نه از وعده سود.',
    'phases': 'ترتیب تأیید: جاروی نقدینگی، پس‌گرفتن سطح، حرکت قوی، تغییر ساختار.',
    'confirmation': 'جهت تایم‌فریم بالاتر برای مقایسه با جهت این تحلیل بررسی می‌شود.',
}
SIDE = {'BUY': 'خرید', 'SELL': 'فروش', 'HOLD': 'صبر؛ بدون معامله'}
STATUS = {'VALID': 'معتبر', 'INVALID': 'نامعتبر', 'FADING': 'رو به پایان',
          'UNKNOWN': 'نامعلوم', 'BREAKING_OUT': 'در حال شکست',
          'APPROACHING_BREAKOUT': 'نزدیک شکست', 'CONSOLIDATING': 'در حال تثبیت',
          'STRONG': 'قوی', 'MEDIUM': 'متوسط', 'WEAK': 'ضعیف', 'VERY_STRONG': 'بسیار قوی'}
PHASE = {'Sweep': 'جاروی نقدینگی', 'Reclaim': 'پس‌گرفتن سطح',
         'Displacement': 'حرکت قوی', 'CHoCH': 'تغییر ساختار'}
DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')


def get_explanation(key: str) -> str:
    return EXPLANATIONS.get(key, 'اطلاعات این بخش در دسترس نیست.')


def _value(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _num(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    if number is None:
        return 'نامعلوم'
    # Keep tiny asset prices readable, rather than rounding valid prices to zero.
    precision = max(digits, 8) if 0 < abs(number) < .01 else digits
    return f'{number:,.{precision}f}'.translate(DIGITS)


def _percent(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and 0 <= number <= 100 else None


def _group_text(group: dict | None, title: str) -> str:
    group = group or {}
    rows = group.get('rows') or []
    if group.get('action') not in ('BUY', 'SELL') or not group.get('hard_filters_passed') or len(rows) != 3:
        reason = group.get('trigger') or '؛ '.join(group.get('hard_filter_reasons') or []) or 'تأیید کافی نداریم.'
        return f'{title}: فعلاً وارد نشوید. {reason} تا این مانع برطرف نشود، این گروه مجوز سفارش نیست.'
    direction = group['action']
    lines = [f"{title}: شرایط {SIDE[direction]} در این گروه تأیید شده؛ قیمت ورود {_num(group.get('entry'))} است."]
    for index, row in enumerate(rows):
        label = _value(row, 'risk_percent', _value(row, 'risk_pct'))
        prefix = f'ریسک {_num(label, 0)}' if label is not None else ('ردیف اول', 'ردیف دوم', 'ردیف سوم')[index]
        entry, stop, target = (_finite(_value(row, key)) for key in ('entry', 'stop_loss', 'take_profit'))
        valid = entry is not None and stop is not None and target is not None and (
            0 < stop < entry < target if direction == 'BUY' else 0 < target < entry < stop)
        if not valid:
            lines.append(f'{prefix}: قیمت‌های معتبر و هم‌جهت موجود نیست؛ این ردیف قابل سفارش نیست.')
            continue
        loss, gain = abs(entry - stop), abs(target - entry)
        probability = _percent(_value(row, 'success_probability'))
        sample_count = _finite(_value(row, 'sample_count'))
        history = (f'در {_num(sample_count, 0)} نمونه تاریخی، {_num(probability)}٪ موفق بوده'
                   if probability is not None and sample_count is not None and sample_count > 0
                   else 'احتمال موفقیت تاریخی معلوم نیست')
        lines.append(f'{prefix}: ورود {_num(entry)}، حدضرر {_num(stop)} و حدسود {_num(target)}؛ '
                     f'به ازای هر یک واحد زیان احتمالی، {_num(gain / loss)} واحد سود هدف‌گذاری شده. {history}.')
    stops = [_finite(_value(row, 'stop_loss')) for row in rows]
    if all(stop is not None for stop in stops) and len(set(stops)) == 1:
        lines.append('در خروجی فعلی حدضرر هر سه ردیف یکسان است؛ بنابراین این جدول سه ریسک متفاوتِ حدضرر را نشان نمی‌دهد.')
    lines.append('این اعداد تضمین سود نیستند؛ حجم معامله و ریسک سرمایه جداگانه تعیین می‌شوند.')
    return '\n'.join(lines)


def build_explanations(result: Any, side: str) -> dict[str, str]:
    if result is None or not _value(result, 'ok', False):
        reason = _value(result, 'reason', '') or 'تحلیل انجام نشده است.'
        return {key: f'فعلاً نمی‌توان درباره این بخش تصمیم گرفت: {reason}' for key in EXPLANATIONS}
    data = (_value(result, 'sides', {}) or {}).get(side, {})
    market = _value(result, 'market', {}) or {}
    symbol = market.get('symbol') or 'این رمزارز'
    out = dict(EXPLANATIONS)
    lines = []
    for name, key in [('مقاومت', 'upper_line'), ('حمایت', 'lower_line')]:
        line = _value(result, key)
        if line is None:
            lines.append(f'خط {name} معتبر پیدا نشده؛ برای تصمیم به این خط تکیه نکنید.')
            continue
        cls = _value(line, 'strength_class', 'UNKNOWN')
        lines.append(f"خط {name} {STATUS.get(cls, cls)} است؛ امتیاز {_num(_value(line, 'strength_score'), 0)} از ۱۰۰، "
                     f"با {_num(_value(line, 'touches'), 0)} برخورد در {_num(_value(line, 'age_candles'), 0)} کندل. "
                     'این سابقه نشان‌دهنده توجه بازار به خط است، نه احتمال قطعی برگشت.')
    out['line_strength'] = '\n'.join(lines)
    pattern = _value(result, 'pattern')
    maturity = _finite(_value(pattern, 'maturity_pct'))
    status = _value(pattern, 'pattern_status', 'UNKNOWN')
    out['pattern'] = (f"برای {symbol} الگوی {_value(pattern, 'pattern_name', 'نامعلوم')} شناسایی شده؛ "
                      f"وضعیتش {STATUS.get(status, status)} و بلوغش {_num(maturity)}٪ است. ")
    if maturity is None:
        out['pattern'] += 'بلوغ معتبر نداریم؛ زمان مناسب ورود از این بخش مشخص نیست.'
    elif maturity > 85:
        out['pattern'] += 'الگو از مرز ۸۵٪ گذشته؛ ورود دیرهنگام تأیید نیست، حتی اگر اسم الگو جذاب باشد.'
    else:
        out['pattern'] += 'از مرز بلوغ ۸۵٪ نگذشته، اما برای ورود باید بقیه شروط هم تأیید شوند.'
    efficiency = _percent(_value(pattern, 'historical_efficiency'))
    out['pattern'] += (' کارایی تاریخی معتبر در این خروجی معلوم نیست.' if efficiency is None else
                       f' کارایی تاریخی گزارش‌شده {_num(efficiency)}٪ است؛ تضمین نتیجه این معامله نیست.')
    levels = _value(result, 'level_info', {}) or {}
    current = _finite(market.get('current_price'))
    descriptions = []
    for key, name, direction in [('resistance_clusters', 'مقاومت', 1), ('support_clusters', 'حمایت', -1)]:
        categories = {'متوسط': [], 'قوی': []}
        for cluster in levels.get(key, []):
            price, touches = _finite(cluster.get('price')), _finite(cluster.get('touches'))
            if price is None or touches is None or touches < 1 or current is None or (price-current)*direction <= 0:
                continue
            categories['قوی' if touches >= 3 else 'متوسط'].append(price)
        for strength, prices in categories.items():
            prices = sorted(set(prices), reverse=direction < 0)[:3]
            descriptions.append(f"{name} {strength}: " + ('، '.join(_num(price) for price in prices) if prices else 'سطح تأییدشده نداریم') + '.')
    out['support_resistance'] = '\n'.join(descriptions) + '\nاین‌ها واکنش‌های ثبت‌شده‌اند، نه قیمت‌های اضافه‌شده برای پرکردن جدول؛ فاصله‌گذاری صفحه ممکن است تعداد نمایشی را کمتر کند.'
    phases = (_value(result, 'phases', {}) or {}).get(side, {})
    done = [PHASE[name] for name in PHASE if phases.get(name, {}).get('passed')]
    missing = [PHASE[name] for name in PHASE if not phases.get(name, {}).get('passed')]
    out['phases'] = ('تا اینجا ' + ('، '.join(done) if done else 'هیچ مرحله‌ای') + ' تأیید شده. ' +
                     ('هنوز ' + '، '.join(missing) + ' باقی مانده؛ صرف لمس سطح کافی نیست، وارد نشوید.' if missing else
                      'هر چهار مرحله ثبت شده، اما فیلترهای دیگر هم باید تأیید شوند.'))
    confirmation = _value(result, 'confirmation', {}) or {}
    tf = confirmation.get('timeframe') or 'بالاتر'
    if confirmation.get('side') == side:
        out['confirmation'] = f'ساختار تایم‌فریم {tf} با جهت {SIDE.get(side, side)} همسو است؛ این شرط برقرار است، نه اینکه سود تضمین شده باشد.'
    else:
        out['confirmation'] = f"همسویی تایم‌فریم {tf} با {SIDE.get(side, side)} تأیید نشده. {confirmation.get('reason') or 'جهت روشن یا هم‌جهت نیست.'}"
    for key, title, section in [('standard', 'ورود با قیمت فعلی', 'standard_box'),
                               ('stophunt', 'سفارش Stop-Hunt', 'stophunt_box'),
                               ('liquidity_sweep', 'گروه جاروی نقدینگی', 'liquidity_sweep')]:
        group = data.get(key) or (_value(result, key + '_signal', {}) or {}).get(side, {})
        out[section] = _group_text(group, title)
        zone = (group or {}).get('sweep_zone')
        if zone is None:
            zone = (group or {}).get('liquidity_zone')
        if section != 'standard_box' and zone is not None:
            lower = _value(zone, 'lower', _value(zone, 'zone_lower'))
            upper = _value(zone, 'upper', _value(zone, 'zone_upper'))
            out[section] += f' محدوده نقدینگی گزارش‌شده از {_num(lower)} تا {_num(upper)} است؛ رسیدن قیمت به آن، به‌تنهایی مجوز ورود نیست.'
    cal = (_value(result, 'calibration', {}) or {}).get(side, {})
    confidence = data.get('confidence') or _value(result, 'confidence')
    score = _finite(_value(confidence, 'total'))
    out['confidence_score'] = (f'امتیاز ساختاری {_num(score)} از ۱۰۰ است.' if score is not None else
                               'امتیاز ساختاری در این خروجی محاسبه نشده؛ آن را با درصد موفقیت اشتباه نگیرید.')
    out['confidence_score'] += f" تعداد نمونه تاریخی این جهت {_num(cal.get('sample_count'), 0)} است. " + (
        'کالیبراسیون گزارش‌شده کافی است، ولی نتیجه آینده تضمین نمی‌شود.' if cal.get('status') == 'CALIBRATED' else
        'کالیبراسیون کافی نیست؛ درصد موفقیت قابل اتکا اعلام نمی‌شود.')
    std = data.get('standard') or (_value(result, 'standard_signal', {}) or {}).get(side, {})
    out['rr'] = (f"کمترین نسبت گزارش‌شده گروه استاندارد ۱ به {_num(std.get('rr'))} است؛ نسبت واقعی هر ردیف از فاصله قیمت‌ها محاسبه می‌شود."
                 if std.get('action') in ('BUY', 'SELL') and std.get('hard_filters_passed') and std.get('rows') else
                 'در این تحلیل نسبت قابل معامله تأیید نشده؛ عدد ثابت ۱ به ۲ نباید جای قیمت‌های واقعی را بگیرد.')
    out['data_source'] = (f"برای {symbol} در تایم‌فریم {market.get('timeframe') or 'نامعلوم'}، منبع {market.get('data_source') or 'نامعلوم'} "
                          f"و {_num(market.get('candles_analyzed'), 0)} کندل بسته استفاده شده؛ قیمت تحلیل {_num(market.get('current_price'))} است.")
    decision = data.get('final_decision', _value(result, 'final_decision', 'HOLD'))
    out['final_decision'] = f"نتیجه این جهت: {SIDE.get(decision, decision)}."
    return out
