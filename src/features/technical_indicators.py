import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free technical features (ratios/returns) so they generalize across price levels."""
    df = df.copy()
    c, v = df["close"], df["volume"]

    df["return_1d"] = c.pct_change()
    df["return_5d"] = c.pct_change(5)

    sma10, sma30 = c.rolling(10).mean(), c.rolling(30).mean()
    df["close_sma10_ratio"] = c / sma10 - 1
    df["close_sma30_ratio"] = c / sma30 - 1

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["ema_diff"] = (ema12 - ema26) / c

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd / c
    df["macd_signal"] = signal / c
    df["macd_hist"] = (macd - signal) / c

    df["volume_change"] = v.pct_change().replace([np.inf, -np.inf], np.nan)
    df["volume_ratio"] = v / v.rolling(20).mean()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    return df
