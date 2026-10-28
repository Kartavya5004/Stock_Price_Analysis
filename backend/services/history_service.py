from datetime import date, timedelta

import pandas as pd

from src.data.stock_fetcher import fetch_stock_data
from src.preprocessing.clean_stock_data import clean_stock_data
from src.utils.errors import InsufficientDataError


def _round(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 2)


def get_history(ticker: str, days: int = 252) -> dict:
    """Last `days` trading bars with 10/30-day moving averages, for the dashboard chart."""
    ticker = ticker.upper()
    # ~1.5 calendar days per trading day, plus a warm-up so the 30-day average exists on the first bar returned
    start = (date.today() - timedelta(days=int(days * 1.5) + 60)).isoformat()
    prices = clean_stock_data(fetch_stock_data(ticker, start=start))
    if len(prices) < 2:
        raise InsufficientDataError(f"Only {len(prices)} price rows for '{ticker}'; nothing to chart.")

    prices["sma10"] = prices["close"].rolling(10).mean()
    prices["sma30"] = prices["close"].rolling(30).mean()
    tail = prices.tail(days)
    points = [
        {"date": d.date().isoformat(), "close": _round(c), "sma10": _round(s10), "sma30": _round(s30)}
        for d, c, s10, s30 in zip(tail["date"], tail["close"], tail["sma10"], tail["sma30"])
    ]
    return {"ticker": ticker, "points": points}
