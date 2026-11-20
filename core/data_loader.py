"""
core/data_loader.py
لایه دریافت و آماده‌سازی داده.

قواعد اصلی طبق سند استراتژی (نسخه اصلاح‌شده):
• منبع اصلی: Yahoo Finance (yfinance)
• منبع دوم: Kraken (OHLC عمومی، بدون کلید)
• منبع سوم: CoinGecko
• ممنوعیت مطلق: هرگونه داده یا API بایننس
• هیچ داده فرضی ساخته نمی‌شود. در صورت کمبود داده -> HOLD + دلیل دقیق.
• تعداد کندل بک‌تست ۲۰۰ تا ۳۰۰ کندل است (طبق توافق نهایی).
• تایم‌فریم تأیید: 15m<-1h ، 1h<-4h ، 4h<-1D ، 1D<-1W
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

# نمایش کاربر -> (نماد یاهو فایننس, شناسه کوین‌گکو, جفت‌ارز کراکن)
CRYPTO_UNIVERSE = {
    "BTC": ("BTC-USD", "bitcoin", "XBTUSD"),
    "ETH": ("ETH-USD", "ethereum", "ETHUSD"),
    "SOL": ("SOL-USD", "solana", "SOLUSD"),
    "BNB": ("BNB-USD", "binancecoin", "BNBUSD"),
    "XRP": ("XRP-USD", "ripple", "XRPUSD"),
    "ADA": ("ADA-USD", "cardano", "ADAUSD"),
    "DOGE": ("DOGE-USD", "dogecoin", "DOGEUSD"),
    "AVAX": ("AVAX-USD", "avalanche-2", "AVAXUSD"),
    "LINK": ("LINK-USD", "chainlink", "LINKUSD"),
    "MATIC": ("MATIC-USD", "matic-network", "MATICUSD"),
    "DOT": ("DOT-USD", "polkadot", "DOTUSD"),
    "LTC": ("LTC-USD", "litecoin", "LTCUSD"),
    "TON": ("TON11419-USD", "the-open-network", "TONUSD"),
    "TRX": ("TRX-USD", "tron", "TRXUSD"),
    "ATOM": ("ATOM-USD", "cosmos", "ATOMUSD"),
}

# ترتیب تایم‌فریم‌ها (برای پیدا کردن «یک مرحله بالاتر» جهت تأیید)
TIMEFRAME_ORDER = ["15m", "1h", "4h", "1D", "1W"]

# فصل ۲.۳ و فصل ۳: ماتریس مقیاس‌بندی تعداد کندل، پنجره پیوت و دوره ATR
# اصلاح شد: تعداد کندل بک‌تست برای همه تایم‌فریم‌ها در بازه ۲۰۰ تا ۳۰۰ قرار گرفت
# (قبلاً برای ۱۵ دقیقه ۹۰۰ کندل بود که کالیبراسیون را مختل می‌کرد).
TIMEFRAME_MATRIX = {
    "15m": dict(yf_interval="15m", base_interval=None, resample=None,
                period="60d", min_candles=200, target_candles=250,
                pivot_window=(20, 40), atr_period=24,
                kraken_interval=15),
    "1h": dict(yf_interval="60m", base_interval=None, resample=None,
               period="730d", min_candles=200, target_candles=280,
               pivot_window=(8, 16), atr_period=14,
               kraken_interval=60),
    "4h": dict(yf_interval="60m", base_interval=None, resample="4h",
               period="730d", min_candles=200, target_candles=260,
               pivot_window=(6, 12), atr_period=14,
               kraken_interval=240),
    "1D": dict(yf_interval="1d", base_interval=None, resample=None,
               period="2y", min_candles=120, target_candles=200,
               pivot_window=(3, 7), atr_period=14,
               kraken_interval=1440),
    "1W": dict(yf_interval="1wk", base_interval=None, resample=None,
               period="10y", min_candles=60, target_candles=120,
               pivot_window=(2, 5), atr_period=8,
               kraken_interval=10080),
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
    # داده تایم‌فریم تأیید (یک مرحله بالاتر) — اصلاح جدید
    confirm_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    confirm_timeframe: str | None = None
    confirm_source: str = "NONE"


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


def _fetch_kraken(pair: str, interval_minutes: int, max_candles: int) -> pd.DataFrame:
    """منبع دوم: Kraken OHLC عمومی (بدون کلید، بدون بایننس)."""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        resp = requests.get(
            url, params={"pair": pair, "interval": interval_minutes}, timeout=15
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            return pd.DataFrame()
        result = payload.get("result", {})
        rows = None
        for key, value in result.items():
            if key != "last" and isinstance(value, list):
                rows = value
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            rows,
            columns=["ts", "Open", "High", "Low", "Close", "vwap", "Volume", "count"],
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("ts")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.iloc[-max_candles:]
        return _clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def _fetch_coingecko(coin_id: str, days: int) -> pd.DataFrame:
    """منبع سوم: CoinGecko OHLC عمومی (دقت کمتر، فقط برای پوشش کمبود داده)."""
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


def _resample_if_needed(df: pd.DataFrame, resample: str | None) -> pd.DataFrame:
    if not resample or df.empty:
        return df
    return (
        df.resample(resample)
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(how="any")
    )


def _fetch_with_fallback(display_symbol: str, timeframe: str) -> tuple[pd.DataFrame, str]:
    """تلاش به ترتیب: Yahoo -> Kraken -> CoinGecko. بایننس هرگز استفاده نمی‌شود."""
    yf_symbol, coin_id, kraken_pair = CRYPTO_UNIVERSE[display_symbol]
    cfg = TIMEFRAME_MATRIX[timeframe]

    # ۱) یاهو فایننس
    df = _fetch_yahoo(yf_symbol, cfg["yf_interval"], cfg["period"])
    df = _resample_if_needed(df, cfg["resample"])
    if not df.empty and len(df) >= cfg["min_candles"]:
        return df, "YAHOO_FINANCE"

    # ۲) کراکن (منبع دوم — اصلاح جدید)
    kr_df = _fetch_kraken(kraken_pair, cfg["kraken_interval"], cfg["target_candles"])
    if not kr_df.empty and len(kr_df) >= cfg["min_candles"]:
        return kr_df, "KRAKEN"

    # ۳) کوین‌گکو (منبع سوم)
    days = 90 if timeframe in ("15m", "1h", "4h") else 365
    cg_df = _fetch_coingecko(coin_id, days)
    if not cg_df.empty and len(cg_df) > len(df):
        return cg_df, "COINGECKO"

    # بهترین داده موجود (حتی اگر ناکافی باشد) همراه با منبعش برگردانده می‌شود
    if not df.empty:
        return df, "YAHOO_FINANCE"
    if not kr_df.empty:
        return kr_df, "KRAKEN"
    if not cg_df.empty:
        return cg_df, "COINGECKO"
    return pd.DataFrame(), "NONE"


def load_market_data(display_symbol: str, timeframe: str) -> MarketData:
    """داده OHLCV را طبق منابع مجاز و ماتریس مقیاس‌بندی زمانی بارگذاری می‌کند."""
    if display_symbol not in CRYPTO_UNIVERSE:
        return MarketData(False, display_symbol, timeframe, reason="رمزارز پشتیبانی نمی‌شود.")
    if timeframe not in TIMEFRAME_MATRIX:
        return MarketData(False, display_symbol, timeframe, reason="تایم‌فریم پشتیبانی نمی‌شود.")

    cfg = TIMEFRAME_MATRIX[timeframe]

    df, source = _fetch_with_fallback(display_symbol, timeframe)

    if df.empty:
        return MarketData(
            False, display_symbol, timeframe, pd.DataFrame(), "NONE", 0, False,
            reason="اطلاعات معتبر برای این بخش موجود نیست و اصلاح انجام نشد.",
        )

    # اگر داده بیش از نیاز است، به تعداد هدف (target_candles) محدود شود
    candles = len(df)
    target = cfg["target_candles"]
    if candles > target:
        df = df.iloc[-target:]
        candles = len(df)

    sufficient = candles >= cfg["min_candles"]
    reason = "" if sufficient else (
        f"تعداد کندل دریافتی ({candles}) کمتر از حداقل لازم "
        f"({cfg['min_candles']}) است؛ نتایج با احتیاط بیشتری تفسیر شوند."
    )

    # --- داده تایم‌فریم تأیید (یک مرحله بالاتر) — اصلاح جدید
    confirm_tf = higher_timeframe(timeframe)
    confirm_df = pd.DataFrame()
    confirm_source = "NONE"
    if confirm_tf is not None:
        confirm_df, confirm_source = _fetch_with_fallback(display_symbol, confirm_tf)

    return MarketData(
        ok=True,
        symbol=display_symbol,
        timeframe=timeframe,
        df=df,
        data_source=source,
        candles_analyzed=candles,
        data_sufficient=sufficient,
        reason=reason,
        confirm_df=confirm_df,
        confirm_timeframe=confirm_tf,
        confirm_source=confirm_source,
    )