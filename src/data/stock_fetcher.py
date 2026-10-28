import pandas as pd
import yfinance as yf

from src.utils.config import RAW_DIR, START_DATE
from src.utils.errors import NoDataError


def fetch_stock_data(ticker: str, start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise NoDataError(f"No price data returned for '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["ticker"] = ticker
    return df


def save_stock_data(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "stock_data.csv", index=False)
