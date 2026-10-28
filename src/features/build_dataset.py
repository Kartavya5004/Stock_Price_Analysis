import numpy as np
import pandas as pd

from src.utils.config import (
    MIN_NEWS_COVERAGE,
    MIN_SENTIMENT_ROWS,
    NEUTRAL_THRESHOLD,
    SENTIMENT_FEATURES,
)

_SENT_MEANS = ["sent_mean", "sent_pos_mean", "sent_neg_mean"]


def align_to_trading_days(
    news: pd.DataFrame, trading_dates: pd.Series, allow_after_last: bool = False
) -> pd.DataFrame:
    """Map each article's date to the first trading day on or after it.

    Weekend/holiday news therefore lands on the next session instead of being dropped.
    Articles later than the last price bar are dropped (no row to attach them to), unless
    allow_after_last=True (live prediction: they belong to the latest known bar).
    """
    if news.empty:
        return news
    cal = (
        pd.DataFrame({"trading_date": pd.to_datetime(pd.Series(trading_dates)).drop_duplicates().sort_values()})
        .astype({"trading_date": "datetime64[ns]"})
        .reset_index(drop=True)
    )
    left = news.copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    left = left.sort_values("date")
    out = pd.merge_asof(left, cal, left_on="date", right_on="trading_date", direction="forward")
    if allow_after_last:
        out["trading_date"] = out["trading_date"].fillna(cal["trading_date"].iloc[-1])
    out = out.dropna(subset=["trading_date"])
    out["date"] = out["trading_date"]
    return out.drop(columns="trading_date").reset_index(drop=True)


def add_sentiment_features(feats: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily sentiment onto price features.

    A day with no articles is *missing* sentiment (NaN, which XGBoost handles natively),
    not neutral sentiment: 0.0 would be indistinguishable from a genuinely balanced news day.
    news_count stays 0 on those days because 'zero articles' is a real observation.
    """
    df = feats.merge(daily_sent, on="date", how="left")
    df["news_count"] = df["news_count"].astype(float).fillna(0.0)
    for col in _SENT_MEANS:
        df[col] = df[col].astype(float)
    df["sent_mean_3d"] = df["sent_mean"].rolling(3, min_periods=1).mean()  # NaN only if 3 empty days
    return df


def add_target(df: pd.DataFrame, threshold: float = NEUTRAL_THRESHOLD) -> pd.DataFrame:
    """Label = direction of NEXT day's return: 0 bearish, 1 neutral, 2 bullish."""
    df = df.copy()
    df["next_return"] = df["close"].pct_change().shift(-1)
    df["target"] = np.select(
        [df["next_return"] > threshold, df["next_return"] < -threshold], [2, 0], default=1
    )
    return df[df["next_return"].notna()]


def build_final_dataset(
    feats: pd.DataFrame, daily_sent: pd.DataFrame, restrict_to_news_window: bool = True
) -> pd.DataFrame:
    """Price features + sentiment + target.

    With news available, rows are limited to [first news day, last news day]. Outside that
    window sentiment is unobserved, and training on it would teach the model that
    'no sentiment' is the normal case. With no news at all the full price history is kept
    (price-only mode).
    """
    df = add_target(add_sentiment_features(feats, daily_sent))
    if restrict_to_news_window and not daily_sent.empty:
        df = df[(df["date"] >= daily_sent["date"].min()) & (df["date"] <= daily_sent["date"].max())]
    required = [c for c in df.columns if c not in SENTIMENT_FEATURES]
    return df.dropna(subset=required).reset_index(drop=True)


def sentiment_coverage(df: pd.DataFrame) -> dict:
    n = len(df)
    days = int((df["news_count"] > 0).sum()) if n and "news_count" in df else 0
    return {"rows": n, "days_with_news": days, "coverage": days / n if n else 0.0}


def sentiment_problems(df: pd.DataFrame) -> list[str]:
    """Reasons the dataset is too thin for a price+sentiment model (empty list = OK)."""
    cov = sentiment_coverage(df)
    problems = []
    if cov["rows"] < MIN_SENTIMENT_ROWS:
        problems.append(f"only {cov['rows']} trading days of data (need >= {MIN_SENTIMENT_ROWS})")
    if cov["coverage"] < MIN_NEWS_COVERAGE:
        problems.append(
            f"only {cov['coverage']:.0%} of days have news "
            f"({cov['days_with_news']}/{cov['rows']}; need >= {MIN_NEWS_COVERAGE:.0%})"
        )
    return problems
