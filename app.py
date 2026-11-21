"""برنامه Streamlit استراتژی هندسی، نقدینگی و Stop-Hunt.
اجرا: streamlit run app.py — فایل‌های اصلاح‌شده باید در پوشه core باشند.
"""
from __future__ import annotations

from html import escape
from math import isfinite
import streamlit as st
from core import data_loader as dl
from core import levels as lv
from core.engine import run_analysis
from core.explanations import EXPLANATIONS, build_explanations

_CURRENT = {}

DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
ACTIONS = {'BUY': 'خرید', 'SELL': 'فروش', 'HOLD': 'صبر؛ بدون معامله'}
STATUS = {'CALIBRATED': 'کالیبره‌شده', 'PARTIALLY_CALIBRATED': 'کالیبراسیون ناقص',
          'NOT_CALIBRATED': 'کالیبره‌نشده', 'STRONG': 'قوی', 'MEDIUM': 'متوسط',
          'WEAK': 'ضعیف', 'UNKNOWN': 'نامعلوم', 'VALID': 'معتبر', 'INVALID': 'نامعتبر',
          'FADING': 'رو به پایان', 'BREAKING_OUT': 'در حال شکست',
          'APPROACHING_BREAKOUT': 'نزدیک شکست', 'CONSOLIDATING': 'تثبیت'}
PHASES = {'Sweep': 'جاروی نقدینگی', 'Reclaim': 'پس‌گرفتن سطح',
          'Displacement': 'حرکت قوی', 'CHoCH': 'تغییر ساختار'}


def fa(value):
    return str(value).translate(DIGITS)


def fmt(value, decimals=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 'نامعلوم'
    if not isfinite(number):
        return 'نامعلوم'
    digits = decimals if decimals is not None else (2 if abs(number) >= 100 else 4 if abs(number) >= 1 else 6)
    return fa(f'{number:,.{digits}f}')


def info_button(key, detail=''):
    with st.expander('ℹ️ توضیح ساده'):
        st.write(fa(_CURRENT.get(key) or EXPLANATIONS.get(key, '')))
        if detail:
            st.write(fa(detail))


def analyze(params):
    return run_analysis(params['symbol'], params['timeframe'], params['side_pref'],
                        params['rr_target'], pivot_sensitivity=params['pivot_sensitivity'],
                        require_htf_confirmation=params['require_htf_confirm'])


def render_settings():
    st.subheader('⚙️ تنظیمات استراتژی')
    # Outside the form: changing timeframe updates its confirmation immediately.
    timeframe = st.selectbox('تایم‌فریم', dl.TIMEFRAME_ORDER, index=dl.TIMEFRAME_ORDER.index('4h'))
    htf = dl.higher_timeframe(timeframe)
    st.caption(f'تایم‌فریم تأیید: {fa(htf)}' if htf else 'تایم‌فریم بالاتری تعریف نشده است.')
    with st.form('settings_form'):
        symbol = st.selectbox('رمزارز', list(dl.CRYPTO_UNIVERSE))
        rr_target = st.selectbox('حداقل نسبت سود به زیان (نه نرخ برد)', [1.5, 2., 2.5, 3.],
                                 index=1, format_func=lambda v: '۱ : ' + fmt(v, 1))
        side_label = st.radio('جهت تحلیل', ['خرید', 'فروش', 'هر دو'], index=2, horizontal=True)
        sensitivity = st.selectbox('حساسیت شناسایی پیوت', ['کم', 'متوسط', 'زیاد'], index=1)
        required = st.checkbox('الزام تأیید هم‌جهت تایم‌فریم بالاتر', value=bool(htf), disabled=not bool(htf))
        submitted = st.form_submit_button('تأیید و اجرای استراتژی', use_container_width=True)
    if submitted:
        params = dict(symbol=symbol, timeframe=timeframe, rr_target=rr_target,
                      side_pref={'خرید': 'BUY', 'فروش': 'SELL', 'هر دو': 'BOTH'}[side_label],
                      pivot_sensitivity={'کم': 'LOW', 'متوسط': 'MEDIUM', 'زیاد': 'HIGH'}[sensitivity],
                      require_htf_confirm=bool(htf) and required)
        try:
            with st.spinner('در حال دریافت داده و تحلیل ساختار قیمت…'):
                result = analyze(params)
        except Exception:
            st.error('دریافت یا تحلیل داده انجام نشد؛ هیچ سیگنالی صادر نشد. دوباره تلاش کنید.')
            return
        st.session_state.params = params
        st.session_state.result = result
        st.session_state.stage = 'results'
        st.rerun()


def price_records(group):
    # HOLD never exposes leftover prices or theoretical Limit entries.
    if group.get('action') not in ('BUY', 'SELL') or not group.get('hard_filters_passed'):
        return []
    rows = group.get('rows') or []
    if len(rows) != 3:
        return []
    records = []
    for index, row in enumerate(rows):
        probability = getattr(row, 'success_probability', None)
        records.append({'سناریو': ('هدف نزدیک', 'هدف میانه', 'هدف دور')[index],
                        'حد سود': fmt(row.take_profit), 'قیمت سفارش': fmt(row.entry),
                        'حدضرر': fmt(row.stop_loss), 'نسبت سود به زیان': fmt(row.rr, 2),
                        'موفقیت تاریخی': fmt(probability, 2) + '٪' if probability is not None else 'نامعلوم',
                        'تعداد نمونه': fmt(getattr(row, 'sample_count', None), 0)})
    return records


def render_price_box(title, side, group, box_key):
    st.subheader(title)
    st.write(ACTIONS.get(group.get('action'), ACTIONS['HOLD']))
    st.caption(fa(group.get('trigger') or 'شواهد کافی موجود نیست.'))
    records = price_records(group)
    if records:
        st.table(records)
    else:
        st.info('قیمت سفارش، حدسود و حدضرر تأیید نشده‌اند.')
    info_button(box_key, group.get('trigger', ''))


def render_side_result(res, side, params):
    data = res.sides.get(side, {})
    _CURRENT.clear()
    try:
        _CURRENT.update(build_explanations(res, side) or {})
    except Exception:
        pass
    market = res.market or {}
    st.subheader(f"{market.get('symbol', '')} / USD — {fmt(market.get('current_price'))} — {fa(market.get('timeframe', ''))}")
    st.write('جهت تحلیل: ' + ACTIONS[side])
    st.write('تصمیم: ' + ACTIONS.get(data.get('final_decision'), ACTIONS['HOLD']))
    if market.get('reason'):
        st.warning(fa(market['reason']))
    st.subheader('۱. قدرت خطوط روند')
    for label, line in [('مقاومت', res.upper_line), ('حمایت', res.lower_line)]:
        if line is None:
            st.write(label + ': نامعلوم')
            continue
        st.write(f"{label}: {STATUS.get(line.strength_class, line.strength_class)} — {fmt(line.strength_score, 0)} / ۱۰۰")
        st.caption(f'برخورد معتبر: {fmt(line.touches, 0)} | زاویه: {fmt(line.angle_degrees, 1)}° | عمر: {fmt(line.age_candles, 0)} کندل')
    info_button('line_strength')
    st.subheader('۲. الگوی کلاسیک')
    pattern = res.pattern
    if pattern is not None:
        st.write(fa(pattern.pattern_name))
        st.write('وضعیت: ' + STATUS.get(pattern.pattern_status, pattern.pattern_status))
        st.write('بلوغ: ' + fmt(pattern.maturity_pct, 1) + '٪')
        efficiency = getattr(pattern, 'historical_efficiency', None)
        st.write('کارایی تاریخی: ' + (fmt(efficiency, 2) + '٪' if efficiency is not None else 'نامعلوم'))
        st.caption('نسبت حجم شکست: ' + fmt(getattr(pattern, 'breakout_volume_ratio', None), 2))
    else:
        st.write('الگو: نامعلوم')
    info_button('pattern')
    st.subheader('۳. حمایت و مقاومت واقعی')
    ladder = lv.build_target_ladder(market.get('current_price'), market.get('atr_value'), side, res.level_info)
    word = 'مقاومت' if side == 'BUY' else 'حمایت'
    for key, label in [('medium', 'متوسط'), ('strong', 'قوی')]:
        values = ladder.get(key, [])
        st.write(f'{word} {label}: ' + ('، '.join(fmt(v) for v in values) if values else 'یافت نشد'))
    info_button('support_resistance')
    st.subheader('۴. چهار مرحله تأیید')
    for key, label in PHASES.items():
        phase = (res.phases or {}).get(side, {}).get(key, {})
        st.write(label + ': ' + ('تأیید شد' if phase.get('passed') else 'تأیید نشده'))
    info_button('phases')
    confirmation = res.confirmation or {}
    st.subheader('۵. تأیید تایم‌فریم بالاتر')
    if not params.get('require_htf_confirm', True):
        st.write('الزام تأیید غیرفعال است.')
    else:
        st.write('هم‌جهت' if confirmation.get('side') == side else 'هم‌جهت تأیید نشده')
    st.caption(fa(confirmation.get('reason', '')))
    info_button('confirmation')
    for key, title, explanation in [
        ('standard', '۶. گروه استاندارد', 'standard_box'),
        ('stophunt', '۷. گروه Stop-Hunt', 'stophunt_box'),
        ('liquidity_sweep', '۸. گروه جاروی نقدینگی', 'liquidity_sweep')]:
        group = data.get(key, {})
        render_price_box(title, side, group, explanation)
        zone = group.get('sweep_zone')
        if zone:
            st.caption('محدوده مشاهده‌شده: ' + fmt(zone.get('lower')) + ' تا ' + fmt(zone.get('upper')))
        # No optimized_limit_price shown while engine holds Limit execution.
    st.subheader('۹. شواهد تاریخی')
    calibration = (res.calibration or {}).get(side, {})
    st.write('وضعیت: ' + STATUS.get(calibration.get('status'), 'نامعلوم'))
    st.write('تعداد نمونه: ' + fmt(calibration.get('sample_count'), 0))
    if data.get('confidence') is None:
        st.write('امتیاز اطمینان ساختاری: نامعلوم')
    info_button('confidence_score')
    if not data.get('hard_filters_passed'):
        st.warning('علت توقف: ' + fa('؛ '.join(data.get('hard_filter_reasons', []) or ['شواهد کافی نیست'])))
    st.caption(f"منبع داده: {market.get('data_source', 'نامعلوم')} | کندل بسته: {fmt(market.get('candles_analyzed'), 0)} | ATR: {fmt(market.get('atr_value'))} | پنجره پیوت: {fmt(market.get('pivot_window'), 0)}")
    info_button('data_source')


def render_results():
    if st.button('بازگشت به تنظیمات', use_container_width=True):
        st.session_state.stage = 'settings'
        st.rerun()
    res = st.session_state.result
    if res is None or not res.ok:
        st.error(fa(getattr(res, 'reason', '') or 'داده معتبر موجود نیست؛ هیچ سیگنالی صادر نشد.'))
        return
    st.subheader('تصمیم نهایی: ' + ACTIONS.get(res.final_decision, ACTIONS['HOLD']))
    sides = [side for side in ('BUY', 'SELL') if side in res.sides]
    if len(sides) == 1:
        render_side_result(res, sides[0], st.session_state.params)
    elif sides:
        tabs = st.tabs(['تحلیل خرید', 'تحلیل فروش'])
        for tab, side in zip(tabs, sides):
            with tab:
                render_side_result(res, side, st.session_state.params)
    else:
        st.info('تحلیل قابل نمایش موجود نیست.')


def main():
    st.set_page_config(page_title='استراتژی هندسی، نقدینگی و Stop-Hunt', page_icon='📈', layout='wide')
    st.markdown('''<style>
    .stApp {direction:rtl; text-align:right; font-family:Tahoma,sans-serif;}
    .block-container {padding-top:1.5rem;}
    [data-testid="stTable"] {overflow-x:auto;}
    </style>''', unsafe_allow_html=True)
    for key, default in [('stage', 'settings'), ('result', None), ('params', {})]:
        if key not in st.session_state:
            st.session_state[key] = default
    st.title('📈 استراتژی جامع تحلیل هندسی، نقدینگی و Stop-Hunt')
    if st.session_state.stage == 'settings':
        render_settings()
    else:
        render_results()
    st.caption('این تحلیل فرضیه احتمالی است، نه توصیه مالی قطعی.')


if __name__ == '__main__':
    main()
