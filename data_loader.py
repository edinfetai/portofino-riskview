from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
    "JPM", "XOM", "JNJ", "PG", "WMT",
    "SPY", "QQQ", "TLT", "GLD", "VNQ",
]

FRED_SERIES = {
    "DGS10": "ten_year_yield",
    "DGS2": "two_year_yield",
    "T10Y2Y": "yield_curve_spread",
    "VIXCLS": "vix",
}


def _safe_filename(text: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in text
    )


def _flatten_yfinance_download(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt die angepassten Schlusskurse aus einem yfinance-Download zurück."""
    if df.empty:
        raise ValueError("yfinance hat keine Marktdaten zurückgegeben.")

    if isinstance(df.columns, pd.MultiIndex):
        level_0 = df.columns.get_level_values(0)

        if "Adj Close" in level_0:
            prices = df["Adj Close"]
        elif "Close" in level_0:
            prices = df["Close"]
        else:
            raise ValueError("In den yfinance-Daten wurden weder 'Adj Close' noch 'Close' gefunden.")

    else:
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]]
        elif "Close" in df.columns:
            prices = df[["Close"]]
        else:
            raise ValueError("In den yfinance-Daten wurden weder 'Adj Close' noch 'Close' gefunden.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)

    return prices


def download_market_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str | None = "2018-01-01",
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Lädt angepasste Schlusskurse.

    Empfohlene Nutzung im Training:
        start="2018-01-01", interval="1d", use_cache=True

    Empfohlene Nutzung in der App:
        period="2y", interval="1d", use_cache=False

    So bleibt das Training reproduzierbar, während die App für jede Analyse
    die neuesten verfügbaren täglichen Daten verwendet.
    """
    tickers = sorted({
        ticker.upper().strip()
        for ticker in tickers
        if ticker and ticker.strip()
    })

    if not tickers:
        raise ValueError("Mindestens ein Marktsymbol ist erforderlich.")

    cache_name = _safe_filename(
        f"prices_{'_'.join(tickers)}_{start}_{end}_{period}_{interval}.csv"
    )
    cache_file = RAW_DIR / cache_name

    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    download_kwargs = {
        "tickers": tickers,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "group_by": "column",
        "threads": True,
    }

    if period:
        download_kwargs["period"] = period
    else:
        download_kwargs["start"] = start
        download_kwargs["end"] = end

    raw = yf.download(**download_kwargs)

    prices = _flatten_yfinance_download(raw)
    prices.columns = [str(col).upper() for col in prices.columns]
    prices = prices[[col for col in prices.columns if col in tickers]]

    if prices.empty:
        raise ValueError("Nach der Ticker-Filterung sind keine nutzbaren Kursdaten vorhanden.")

    if use_cache:
        prices.to_csv(cache_file)

    return prices


def _download_single_fred_series(
    series_id: str,
    column_name: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Lädt eine einzelne FRED-Zeitreihe über den öffentlichen CSV-Endpunkt."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    df = pd.read_csv(url)
    df = df.rename(columns={"observation_date": "date", series_id: column_name})

    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    df = df.dropna(subset=[column_name])
    df = df.set_index("date").sort_index()

    df = df.loc[df.index >= pd.Timestamp(start)]

    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]

    return df[[column_name]]


def download_fred_macro(
    start: str = "2018-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Lädt ausgewählte Makroindikatoren aus FRED."""
    cache_name = _safe_filename(f"fred_macro_{start}_{end or 'latest'}.csv")
    cache_file = RAW_DIR / cache_name

    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    frames = []

    for series_id, column_name in FRED_SERIES.items():
        try:
            frames.append(
                _download_single_fred_series(
                    series_id=series_id,
                    column_name=column_name,
                    start=start,
                    end=end,
                )
            )
        except Exception:
            continue

    if not frames:
        raise ValueError("Es konnten keine FRED-Makrodaten geladen werden.")

    macro = pd.concat(frames, axis=1).sort_index()
    macro = macro.ffill().bfill()

    if use_cache:
        macro.to_csv(cache_file)

    return macro


def data_freshness_summary(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    market_source: str,
    interval: str,
) -> dict:
    """Erzeugt eine kompakte Übersicht über Aktualität und Umfang der verwendeten Daten."""
    latest_price_timestamp = prices.dropna(how="all").index.max()
    latest_macro_timestamp = macro.dropna(how="all").index.max()

    return {
        "market_source": market_source,
        "price_interval": interval,
        "latest_price_timestamp": str(latest_price_timestamp),
        "latest_macro_timestamp": str(latest_macro_timestamp),
        "number_of_price_rows": int(len(prices)),
        "number_of_price_columns": int(len(prices.columns)),
        "note": (
            "Das Modell wurde auf täglichen Features trainiert. "
            "Die App verwendet deshalb standardmässig die neuesten verfügbaren täglichen Marktdaten."
        ),
    }
