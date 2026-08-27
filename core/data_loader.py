"""
core/data_loader.py
--------------------
لایه دریافت و آماده‌سازی داده.

قواعد اصلی طبق سند استراتژی:
- منبع اصلی: Yahoo Finance (yfinance)
- منبع پشتیبان: CoinGecko
- ممنوعیت مطلق: هرگونه داده یا API بایننس
- هیچ داده فرضی ساخته نمی‌شود. در صورت کمبود داده -> HOLD + دلیل دقیق.
- تعداد کندل بر اساس افق زمانی واقعی محاسبه می‌شود، نه عدد ثابت.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# ---------------------------------------------------------------------------
# نمادها
# ---------------------------------------------------------------------------

# نمایش کاربر -> (نماد یاهو فایننس, شناسه کوین‌گکو)
CRYPTO_UNIVERSE = {
    "BTC": ("BTC-USD", "bitcoin"),
    "ETH": ("ETH-USD", "ethereum"),
    "SOL": ("SOL-USD", "solana"),
    "BNB": ("BNB-USD", "binancecoin"),
    "XRP": ("XRP-USD", "ripple"),
    "ADA": ("ADA-USD", "cardano"),
    "DOGE": ("DOGE-USD", "dogecoin"),
    "AVAX": ("AVAX-USD", "avalanche-2"),
    "LINK": ("LINK-USD", "chainlink"),
    "MATIC": ("MATIC-USD", "matic-network"),
    "DOT": ("DOT-USD", "polkadot"),
    "LTC": ("LTC-USD", "litecoin"),
    "TON": ("TON11419-USD", "the-open-network"),
    "TRX": ("TRX-USD", "tron"),
    "ATOM": ("ATOM-USD", "cosmos"),
}

# ترتیب تایم‌فریم‌ها (برای پیدا کردن «یک مرحله بالاتر» جهت تأیید)
TIMEFRAME_ORDER = ["15m", "1h", "4h", "1D", "1W"]

# فصل ۲.۳ و فصل ۳: ماتریس مقیاس‌بندی تعداد کندل، پنجره پیوت و دوره ATR
TIMEFRAME_MATRIX = {
    "15m": dict(yf_interval="15m", base_interval=None, resample=None,
                period="60d", min_candles=400, target_candles=900,
                pivot_window=(20, 40), atr_period=24),
    "1h":  dict(yf_interval="60m", base_interval=None, resample=None,
                period="730d", min_candles=150, target_candles=350,
                pivot_window=(8, 16), atr_period=14),
    "4h":  dict(yf_interval="60m", base_interval=None, resample="4h",
                period="730d", min_candles=150, target_candles=300,
                pivot_window=(6, 12), atr_period=14),
    "1D":  dict(yf_interval="1d", base_interval=None, resample=None,
                period="2y", min_candles=45, target_candles=120,
                pivot_window=(3, 7), atr_period=14),
    "1W":  dict(yf_interval="1wk", base_interval=None, resample=None,
                period="10y", min_candles=20, target_candles=60,
                pivot_window=(2, 5), atr_period=8),
}


def higher_timeframe(tf: str) -> str | None:
    """یک تایم‌فریم بالاتر جهت تأیید ساختاری برمی‌گرداند."""
    if tf not in TIMEFRAME_ORDER:
        return None
    idx = TIMEFRAME_ORDER.index(tf)
    if idx + 1 < len(TIMEFRAME_ORDER):
        return TIMEFRAME_ORDER[idx + 1]
    return None


@dataclass
class MarketData:
    ok: bool
    symbol: str
    timeframe: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_source: str = "NONE"
    candles_analyzed: int = 0
    data_sufficient: bool = False
    reason: str = ""


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="any")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_yahoo(yf_symbol: str, interval: str, period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        raw = yf.download(
            yf_symbol, interval=interval, period=period,
            progress=False, auto_adjust=False, threads=False,
        )
        return _clean_ohlcv(raw)
    except Exception:
        return pd.DataFrame()


def _fetch_coingecko(coin_id: str, days: int) -> pd.DataFrame:
    """پشتیبان: CoinGecko OHLC عمومی (دقت کمتر از یاهو، فقط برای پوشش کمبود داده)."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        resp = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        df["Volume"] = np.nan
        return _clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def load_market_data(display_symbol: str, timeframe: str) -> MarketData:
    """داده OHLCV را طبق منابع مجاز و ماتریس مقیاس‌بندی زمانی بارگذاری می‌کند."""
    if display_symbol not in CRYPTO_UNIVERSE:
        return MarketData(False, display_symbol, timeframe, reason="رمزارز پشتیبانی نمی‌شود.")
    if timeframe not in TIMEFRAME_MATRIX:
        return MarketData(False, display_symbol, timeframe, reason="تایم‌فریم پشتیبانی نمی‌شود.")

    yf_symbol, coin_id = CRYPTO_UNIVERSE[display_symbol]
    cfg = TIMEFRAME_MATRIX[timeframe]

    df = _fetch_yahoo(yf_symbol, cfg["yf_interval"], cfg["period"])
    source = "YAHOO_FINANCE"

    if cfg["resample"]:
        if not df.empty:
            df = (
                df.resample(cfg["resample"])
                .agg({"Open": "first", "High": "max", "Low": "min",
                      "Close": "last", "Volume": "sum"})
                .dropna(how="any")
            )

    if df.empty or len(df) < cfg["min_candles"]:
        # تلاش برای پشتیبان CoinGecko (فقط روزانه/ساعتی تقریبی)
        days = 90 if timeframe in ("15m", "1h", "4h") else 365
        cg_df = _fetch_coingecko(coin_id, days)
        if not cg_df.empty and len(cg_df) > len(df):
            df = cg_df
            source = "COINGECKO"

    candles = len(df)
    sufficient = candles >= cfg["min_candles"]

    if df.empty:
        return MarketData(
            False, display_symbol, timeframe, pd.DataFrame(), "NONE", 0, False,
            reason="اطلاعات معتبر برای این بخش موجود نیست و اصلاح انجام نشد.",
        )

    # اگر داده بیش از نیاز است، به تعداد هدف (target_candles) محدود شود
    target = cfg["target_candles"]
    if candles > target:
        df = df.iloc[-target:]
        candles = len(df)

    reason = "" if sufficient else (
        f"تعداد کندل دریافتی ({candles}) کمتر از حداقل لازم "
        f"({cfg['min_candles']}) است؛ نتایج با احتیاط بیشتری تفسیر شوند."
    )

    return MarketData(
        ok=True,
        symbol=display_symbol,
        timeframe=timeframe,
        df=df,
        data_source=source,
        candles_analyzed=candles,
        data_sufficient=sufficient,
        reason=reason,
    )
