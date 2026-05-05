"""Data loading and caching. Sprint 1: yfinance daily only."""

from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "data" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_daily(symbol, start="2010-01-01", end=None, refresh=False):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    cache_path = CACHE_DIR / f"{symbol}_daily.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        df = df.loc[start:end]
        if len(df) > 0:
            _validate(df, symbol)
            return df

    print(f"Downloading {symbol} from yfinance: {start} to {end}")
    raw = yf.download(symbol, start=start, end=end,
                      auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"

    _validate(df, symbol)
    df.to_parquet(cache_path)
    return df


def _validate(df, symbol):
    if df.isna().any().any():
        raise ValueError(f"{symbol}: NaN values present")
    if (df["Volume"] == 0).any():
        n = (df["Volume"] == 0).sum()
        print(f"WARNING: {symbol} has {n} zero-volume days")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: timestamps not monotonic")
    if (df["High"] < df["Low"]).any():
        raise ValueError(f"{symbol}: High < Low")
    if (df["Close"] <= 0).any():
        raise ValueError(f"{symbol}: non-positive close")
