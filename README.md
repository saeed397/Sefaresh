# 📈 استراتژی جامع تحلیل هندسی، نقدینگی و Stop-Hunt

پیاده‌سازی پایتون/Streamlit استراتژی «تحلیل خط روند، الگوهای کلاسیک، حمایت و مقاومت و شکار حدضررها»
بر اساس سند تخصصی ارائه‌شده. این اپلیکیشن دو سیگنال کاملاً مستقل تولید می‌کند:

1. **سیگنال استاندارد (فوری)** — بر اساس تأیید ساختاری قیمت لحظه‌ای.
2. **سیگنال Stop-Hunt (لیمیت)** — سفارش معلق روی ناحیه احتمالی تجمع نقدینگی.

## ⚠️ سلب مسئولیت
این ابزار صرفاً یک موتور تحلیل فنی است و **توصیه مالی قطعی محسوب نمی‌شود**. تمام خروجی‌ها،
به‌ویژه بخش Stop-Hunt، فرضیه‌های احتمالی هستند، نه پیش‌بینی قطعی. مسئولیت هر تصمیم معاملاتی
بر عهدهٔ کاربر است.

## ساختار پروژه
```
trading-strategy-app/
├── app.py                  # رابط کاربری Streamlit (تنظیمات + خروجی سیگنال)
├── requirements.txt
├── .streamlit/config.toml  # تم رنگی برنامه
├── core/
│   ├── data_loader.py      # دریافت داده از Yahoo Finance (پشتیبان: CoinGecko)
│   ├── indicators.py       # Pivot، خط روند، قدرت خط، ATR
│   ├── patterns.py         # شناسایی و اعتبارسنجی الگوهای کلاسیک
│   ├── levels.py           # سطوح حمایت/مقاومت و نردبان قیمتی
│   ├── liquidity.py        # ناحیه نقدینگی و کالیبراسیون Stop-Hunt
│   ├── signals.py          # سیگنال‌ها، مدیریت ریسک، امتیاز اطمینان
│   ├── explanations.py     # متن دکمه‌های ℹ️ (ساده و عامیانه)
│   └── engine.py           # ارکستراسیون کامل پایپ‌لاین
└── README.md
```

## اجرای محلی
```bash
python -m venv .venv
source .venv/bin/activate        # ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## آپلود در GitHub
```bash
git init
git add .
git commit -m "Initial commit: comprehensive trend/pattern/liquidity strategy"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

## اتصال به Streamlit Community Cloud
1. وارد https://share.streamlit.io شوید و با حساب GitHub وارد شوید.
2. روی **New app** بزنید و ریپازیتوری بالا را انتخاب کنید.
3. مسیر فایل اصلی را `app.py` قرار دهید (شاخه `main`).
4. روی **Deploy** بزنید؛ Streamlit به‌صورت خودکار `requirements.txt` را نصب می‌کند.

## منابع داده
- منبع اصلی: [Yahoo Finance](https://finance.yahoo.com) از طریق کتابخانه `yfinance`
- منبع پشتیبان: [CoinGecko](https://www.coingecko.com) (در صورت کمبود داده از یاهو)
- طبق الزام سند استراتژی، **هیچ داده یا API بایننس استفاده نمی‌شود**.

## منطق کلی پایپ‌لاین
```
Market Context & Data Validation
        ↓
Pivot & Trendline Construction
        ↓
Classic Pattern Recognition & Validation
        ↓
Static Support / Resistance Detection
        ↓
Liquidity Zone Identification
        ↓
Stop-Hunt Hypothesis & Sweep Calibration
        ↓
Dual Signal Generation (Immediate + Limit)
        ↓
Stop Loss Model (Structural + Anti-Hunt)
        ↓
Take Profit Model (Geometric + Static Confluence)
        ↓
Risk & Confidence Filter (Hard Filters + Score)
        ↓
Final Decision & Order Preparation
```

## نکات پیاده‌سازی مهم
- تمام آستانه‌ها و پارامترها (پنجره Pivot، دوره ATR، تعداد کندل) بر اساس **ماتریس مقیاس‌بندی
  زمانی** فصل ۳ سند محاسبه می‌شوند و بین تایم‌فریم‌ها ثابت نیستند.
- ضریب کالیبراسیون Stop-Hunt در صورت وجود حداقل ۳۰ رخداد تاریخی معتبر «CALIBRATED» می‌شود؛
  در غیر این صورت از بازه‌های عمومی مرجع (فصل ۱۱.۳) استفاده و وضعیت آن شفاف گزارش می‌شود.
- در صورت کمبود داده، سیستم هرگز داده فرضی نمی‌سازد و پیام
  «اطلاعات معتبر برای این بخش موجود نیست و اصلاح انجام نشد» نمایش داده می‌شود.
- فیلترهای سخت (Hard Filters) حتی با امتیاز اطمینان بالا نیز می‌توانند سیگنال را لغو کنند.

## توسعه‌های پیشنهادی بعدی
- افزودن ماژول ثبت معاملات (Performance Logging) و پایگاه‌داده برای Walk-Forward واقعی.
- افزودن رسم نمودار کندل‌استیک تعاملی (Plotly) برای نمایش بصری خطوط و سطوح.
- کش کردن داده‌های دریافتی برای کاهش تعداد درخواست به Yahoo Finance.
