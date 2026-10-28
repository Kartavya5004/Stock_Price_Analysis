import pandas as pd

from src.utils.config import MARKET_CLOSE_HOUR, MARKET_TZ

_TZ_SUFFIX = r"(?:Z|[+-]\d{2}:?\d{2})$"


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, drop empties/duplicates, build a single 'text' column.

    The output 'date' is the *market date the article is usable for*, assuming
    predictions are made at the close:
      - timezone-aware timestamps (e.g. NewsAPI's '...Z') are converted to US/Eastern;
      - naive timestamps / date-only values are taken as US/Eastern wall-clock time;
      - anything published at or after the close (16:00 ET) moves to the next calendar day,
        because the close-time features could not have seen it.
    Weekend/holiday dates are mapped to a trading day later by align_to_trading_days().
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "text"])
    df = df.copy()

    raw = df["date"].astype(str)
    has_tz = raw.str.contains(_TZ_SUFFIX)
    ts_utc = pd.to_datetime(df["date"], utc=True, errors="coerce", format="mixed")
    eastern = ts_utc.dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    wall = eastern.where(has_tz, ts_utc.dt.tz_localize(None))  # naive values: keep wall-clock as given

    after_close = (wall.dt.hour >= MARKET_CLOSE_HOUR).astype(int)
    df["date"] = wall.dt.normalize() + pd.to_timedelta(after_close, unit="D")

    title = df["title"].fillna("")
    desc = df["description"].fillna("") if "description" in df else ""
    df["text"] = (title + ". " + desc).str.strip(". ").str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["text"].str.len() > 10].drop_duplicates("text")
    return df.sort_values("date").reset_index(drop=True)
