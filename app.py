"""
app.py
------
اپلیکیشن Streamlit استراتژی جامع تحلیل خط روند، الگوهای کلاسیک، حمایت/مقاومت و Stop-Hunt.

اجرا:
    streamlit run app.py

آماده برای:
    - آپلود در یک ریپازیتوری GitHub
    - اتصال مستقیم به Streamlit Community Cloud
"""

from __future__ import annotations

import streamlit as st

from core import data_loader as dl
from core.engine import run_analysis
from core.explanations import EXPLANATIONS

st.set_page_config(
    page_title="استراتژی هندسی، نقدینگی و Stop-Hunt",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# استایل کلی
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { direction: rtl; text-align: right; font-family: 'Vazirmatn', Tahoma, sans-serif; }
    .block-container { padding-top: 1.5rem; }

    .settings-box {
        background: linear-gradient(145deg, #101826, #16233a);
        border: 1px solid #2c3e57;
        border-radius: 18px;
        padding: 28px 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }
    .settings-title {
        font-size: 22px; font-weight: 800; color: #f5f7fa; margin-bottom: 4px;
    }
    .settings-sub { color: #93a3bb; font-size: 14px; margin-bottom: 20px; }

    .signal-header {
        border-radius: 16px; padding: 18px 24px; display: flex; align-items: center;
        justify-content: space-between; font-weight: 800; font-size: 22px; color: white;
        margin-bottom: 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .signal-buy { background: linear-gradient(120deg,#0f5132,#12b76a); }
    .signal-sell { background: linear-gradient(120deg,#5c1010,#e5484d); }

    .metric-card {
        background: #131c2b; border: 1px solid #253552; border-radius: 14px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .lvl-strong { color: #12b76a; font-weight: 800; }
    .lvl-medium { color: #f5a623; font-weight: 700; }

    .box-title { font-size: 17px; font-weight: 800; color: #eef2f8; margin: 6px 0 12px 0;
        border-right: 4px solid #3b82f6; padding-right: 10px; }

    .tp-cell { color: #12b76a; font-weight: 800; }
    .entry-cell { color: #f5c211; font-weight: 800; }
    .sl-cell { color: #ef4444; font-weight: 800; }

    .pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12px;
        font-weight: 700; margin-left: 6px; }
    .pill-strong { background: rgba(18,183,106,0.15); color:#12b76a; border:1px solid rgba(18,183,106,0.4);}
    .pill-medium { background: rgba(245,166,35,0.15); color:#f5a623; border:1px solid rgba(245,166,35,0.4);}
    .pill-weak { background: rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.4);}

    .footer-note { color:#7c8aa0; font-size: 12px; margin-top: 24px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "stage" not in st.session_state:
    st.session_state.stage = "settings"
if "result" not in st.session_state:
    st.session_state.result = None
if "params" not in st.session_state:
    st.session_state.params = {}


def fmt(price: float | None, decimals: int | None = None) -> str:
    if price is None:
        return "—"
    if decimals is None:
        decimals = 2 if price >= 100 else (4 if price >= 1 else 6)
    return f"{price:,.{decimals}f}"


def strength_pill(cls: str) -> str:
    mapping = {"STRONG": ("pill-strong", "قوی"), "MEDIUM": ("pill-medium", "متوسط"),
               "WEAK": ("pill-weak", "ضعیف")}
    css_cls, label = mapping.get(cls, ("pill-weak", cls))
    return f'<span class="pill {css_cls}">{label}</span>'


def info_button(key: str, label: str = "ℹ️ توضیح ساده"):
    with st.expander(label, expanded=False):
        st.write(EXPLANATIONS.get(key, ""))


# ---------------------------------------------------------------------------
# مرحله ۱: صفحه تنظیمات
# ---------------------------------------------------------------------------
def render_settings():
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">⚙️ تنظیمات استراتژی</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-sub">مقادیر پیش‌فرض بهترین تنظیمات پیشنهادی هستند؛ در صورت نیاز تغییر دهید.</div>',
        unsafe_allow_html=True,
    )

    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("۱. انتخاب رمزارز", list(dl.CRYPTO_UNIVERSE.keys()), index=0)
        with col2:
            timeframe = st.selectbox(
                "۲. انتخاب تایم‌فریم", dl.TIMEFRAME_ORDER,
                index=dl.TIMEFRAME_ORDER.index("4h"),
            )

        htf = dl.higher_timeframe(timeframe)
        if htf:
            st.caption(f"🔺 تایم‌فریم تأیید (یک مرحله بالاتر، به‌صورت خودکار): **{htf}** — "
                       "توصیه می‌شود سیگنال نهایی با این تایم‌فریم نیز هم‌راستا باشد.")
        else:
            st.caption("این بالاترین تایم‌فریم موجود است؛ تأیید بالاتر لازم نیست.")

        col3, col4 = st.columns(2)
        with col3:
            rr_label = st.selectbox(
                "۳. حداقل نسبت وین‌ریت (R:R)",
                ["1 : 1.5 (محافظه‌کارانه)", "1 : 2 (پیشنهاد پیش‌فرض)", "1 : 2.5", "1 : 3 (تهاجمی)"],
                index=1,
            )
            rr_target = float(rr_label.split(":")[1].split(" ")[1])
        with col4:
            side_label = st.radio(
                "۴. جهت سیگنال", ["🟢 Buy", "🔴 Sell", "هر دو (Buy & Sell)"], index=2,
                horizontal=True,
            )
            side_map = {"🟢 Buy": "BUY", "🔴 Sell": "SELL", "هر دو (Buy & Sell)": "BOTH"}
            side_pref = side_map[side_label]

        st.markdown("---")
        st.markdown("**تنظیمات تخصصی پیشنهادی (قابل تغییر توسط شما):**")
        col5, col6 = st.columns(2)
        with col5:
            require_htf_confirm = st.checkbox(
                "الزام تأیید ساختاری تایم‌فریم بالاتر پیش از اجرای فوری", value=False,
                help="در صورت فعال بودن، توصیه می‌شود سیگنال فوری فقط زمانی اجرا شود که جهت در تایم‌فریم بالاتر نیز هم‌سو باشد.",
            )
        with col6:
            pivot_sensitivity = st.select_slider(
                "حساسیت شناسایی نقاط چرخش (Pivot)",
                options=["کم (خطوط بلندمدت‌تر)", "متوسط (پیش‌فرض پیشنهادی)", "زیاد (واکنش سریع‌تر)"],
                value="متوسط (پیش‌فرض پیشنهادی)",
            )

        submitted = st.form_submit_button("✅ تأیید و اجرای استراتژی", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        st.session_state.params = {
            "symbol": symbol, "timeframe": timeframe, "rr_target": rr_target,
            "side_pref": side_pref, "require_htf_confirm": require_htf_confirm,
            "pivot_sensitivity": pivot_sensitivity,
        }
        with st.spinner("در حال دریافت داده و تحلیل ساختار قیمت..."):
            result = run_analysis(symbol, timeframe, side_pref, rr_target)
        st.session_state.result = result
        st.session_state.stage = "results"
        st.rerun()


# ---------------------------------------------------------------------------
# مرحله ۲: صفحه سیگنال خروجی
# ---------------------------------------------------------------------------
def render_price_box(title: str, side: str, rows, box_key: str):
    st.markdown(f'<div class="box-title">{title}</div>', unsafe_allow_html=True)
    header_cols = st.columns([1, 1, 1, 1])
    if side == "BUY":
        header_cols[0].markdown('**🟢 حد سود**')
        header_cols[1].markdown('**قیمت سفارش**')
        header_cols[2].markdown('**🔴 حدضرر**')
    else:
        header_cols[0].markdown('**🔴 حد سود**')
        header_cols[1].markdown('**قیمت سفارش**')
        header_cols[2].markdown('**🟢 حدضرر**')
    header_cols[3].markdown('**درصد ریسک**')

    for row in rows:
        c = st.columns([1, 1, 1, 1])
        c[0].markdown(f'<span class="tp-cell">{fmt(row.take_profit)}</span>', unsafe_allow_html=True)
        c[1].markdown(f'<span class="entry-cell">{fmt(row.entry)}</span>', unsafe_allow_html=True)
        c[2].markdown(f'<span class="sl-cell">{fmt(row.stop_loss)}</span>', unsafe_allow_html=True)
        c[3].markdown(f'{row.risk_pct} درصد &nbsp; (R:R ≈ 1:{row.rr:.2f})', unsafe_allow_html=True)

    info_button(box_key)


def render_side_result(res, side: str, params: dict):
    side_data = res.sides[side]
    market = res.market
    price = market["current_price"]
    symbol = market["symbol"]
    tf = market["timeframe"]

    emoji = "🟢" if side == "BUY" else "🔴"
    header_cls = "signal-buy" if side == "BUY" else "signal-sell"
    action_fa = side_data["final_decision"]

    st.markdown(
        f'<div class="signal-header {header_cls}">'
        f'<span>{emoji} &nbsp; {symbol}/USDT &nbsp;&nbsp; {fmt(price)} &nbsp;&nbsp; '
        f'{"Buy" if side=="BUY" else "Sell"} &nbsp;&nbsp; {tf}</span>'
        f'<span style="font-size:15px; background:rgba(255,255,255,0.15); padding:6px 14px; border-radius:10px;">'
        f'{action_fa}</span></div>',
        unsafe_allow_html=True,
    )

    if market["reason"]:
        st.warning(market["reason"])

    # --- قدرت خطوط
    up, low = res.upper_line, res.lower_line
    st.markdown('<div class="box-title">۱. قدرت خطوط روند</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f"**خط مقاومت (بالایی)** {strength_pill(up.strength_class)}", unsafe_allow_html=True)
        st.progress(min(1.0, up.strength_score / 100), text=f"{up.strength_score:.0f} / 100")
        st.caption(f"برخورد معتبر: {up.touches} | زاویه: {up.angle_degrees:.1f}° | "
                   f"عمر: {up.age_candles} کندل | سرعت دورشدن: {up.velocity}")
    with lc2:
        st.markdown(f"**خط حمایت (پایینی)** {strength_pill(low.strength_class)}", unsafe_allow_html=True)
        st.progress(min(1.0, low.strength_score / 100), text=f"{low.strength_score:.0f} / 100")
        st.caption(f"برخورد معتبر: {low.touches} | زاویه: {low.angle_degrees:.1f}° | "
                   f"عمر: {low.age_candles} کندل | سرعت دورشدن: {low.velocity}")
    info_button("line_strength")

    # --- الگوی کلاسیک
    p = res.pattern
    st.markdown('<div class="box-title">۲. الگوی کلاسیک شناسایی‌شده</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("نام الگو", p.pattern_name)
    pc2.metric("وضعیت", p.pattern_status)
    pc3.metric("بلوغ الگو", f"{p.maturity_pct:.0f}%")
    pc4.metric("کارایی تاریخی", f"{p.historical_efficiency}%" if p.historical_efficiency else "نمونه ناکافی")
    st.caption(f"روند حجم داخل الگو: {p.volume_trend} | نسبت حجم شکست: {p.breakout_volume_ratio:.2f}x | "
               f"تقارن زمانی: {p.temporal_symmetry_pct:.0f}% | فضای خالی: {p.white_space} ({p.white_space_quality})")
    info_button("pattern")

    # --- سطوح حمایت/مقاومت
    ladder = res.ladder_buy if side == "BUY" else res.ladder_sell
    title_word = "مقاومت" if side == "BUY" else "حمایت"
    st.markdown(f'<div class="box-title">۳. سه {title_word} متوسط و سه {title_word} قوی</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"**{title_word} متوسط**")
        for v in ladder.get("medium", []):
            st.markdown(f'<div class="metric-card">{fmt(v)}</div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f"**{title_word} قوی**")
        for v in ladder.get("strong", []):
            st.markdown(f'<div class="metric-card lvl-strong">{fmt(v)}</div>', unsafe_allow_html=True)
    info_button("support_resistance")

    # --- گروه ۱: استاندارد
    st.markdown("### ۴. گروه ۱ — قیمت‌گذاری استاندارد (بر اساس قیمت لحظه‌ای)")
    st.caption(side_data["standard"]["trigger"])
    render_price_box("قیمت‌گذاری استاندارد", side, side_data["standard"]["rows"], "standard_box")

    # --- گروه ۲: استاپ‌هانت
    st.markdown("### ۵. گروه ۲ — قیمت‌گذاری Stop-Hunt (سفارش معلق)")
    cal = side_data["stophunt"]["calibration"]
    zone = side_data["stophunt"]["liquidity_zone"]
    st.caption(side_data["stophunt"]["trigger"])
    zc1, zc2, zc3 = st.columns(3)
    zc1.metric("محدوده نقدینگی (پایین)", fmt(zone.zone_lower))
    zc2.metric("نقطه بهینه سفارش Limit", fmt(cal.optimized_limit_price))
    zc3.metric("محدوده نقدینگی (بالا)", fmt(zone.zone_upper))
    st.caption(f"وضعیت کالیبراسیون: {cal.calibration_status} | نمونه‌های تاریخی: {cal.sample_size} | "
               f"ضریب نوسان اختصاصی: {cal.backtest_volatility_multiplier:.2f}× ATR")
    render_price_box("قیمت‌گذاری Stop-Hunt", side, side_data["stophunt"]["rows"], "stophunt_box")

    # --- امتیاز اطمینان
    conf = side_data["confidence"]
    st.markdown('<div class="box-title">۶. امتیاز اطمینان ساختاری (غیرذهنی)</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.metric("امتیاز کل", f"{conf.total:.0f} / 100", conf.score_class)
    with cc2:
        st.progress(min(1.0, conf.total / 100))
        st.caption(" | ".join(f"{k}: {v}" for k, v in conf.components.items()))
    info_button("confidence_score")

    if not side_data["hard_filters_passed"]:
        st.error("فیلترهای سخت رد شدند: " + "، ".join(side_data["hard_filter_reasons"]))

    st.markdown(
        f'<div class="footer-note">منبع داده: {market["data_source"]} | تعداد کندل تحلیل‌شده: '
        f'{market["candles_analyzed"]} | دوره ATR: {market["atr_period"]} کندل | '
        f'پنجره Pivot: {market["pivot_window"]} کندل | این تحلیل فرضیه احتمالی است، نه توصیه مالی قطعی.</div>',
        unsafe_allow_html=True,
    )


def render_results():
    res = st.session_state.result
    params = st.session_state.params

    top_l, top_r = st.columns([5, 1])
    with top_r:
        if st.button("⚙️ بازگشت به تنظیمات", use_container_width=True):
            st.session_state.stage = "settings"
            st.rerun()

    if not res.ok:
        st.error(res.reason or "اطلاعات معتبر برای این بخش موجود نیست و اصلاح انجام نشد.")
        return

    sides = list(res.sides.keys())
    if len(sides) == 1:
        render_side_result(res, sides[0], params)
    else:
        tabs = st.tabs(["🟢 سیگنال Buy", "🔴 سیگنال Sell"])
        with tabs[0]:
            render_side_result(res, "BUY", params)
        with tabs[1]:
            render_side_result(res, "SELL", params)


# ---------------------------------------------------------------------------
# روتینگ
# ---------------------------------------------------------------------------
st.title("📈 استراتژی جامع تحلیل هندسی، نقدینگی و Stop-Hunt")

if st.session_state.stage == "settings":
    render_settings()
else:
    render_results()
