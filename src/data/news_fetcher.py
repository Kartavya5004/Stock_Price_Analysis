import warnings
from datetime import date, timedelta

import pandas as pd
import requests

from src.utils.config import NEWS_API_KEY, RAW_DIR

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news(query: str, days: int = 28) -> pd.DataFrame:
    """Recent headlines from NewsAPI. The free tier only reaches back ~30 days,
    so use load_news_csv() with a historical dataset for training."""
    cols = ["date", "title", "description", "source"]
    if not NEWS_API_KEY:
        warnings.warn("NEWS_API_KEY not set; continuing without news.")
        return pd.DataFrame(columns=cols)

    params = {
        "q": query,
        "from": (date.today() - timedelta(days=days)).isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=30)
    resp.raise_for_status()
    rows = [
        {
            "date": a["publishedAt"],
            "title": a.get("title"),
            "description": a.get("description"),
            "source": (a.get("source") or {}).get("name"),
        }
        for a in resp.json().get("articles", [])
    ]
    return pd.DataFrame(rows, columns=cols)


def load_news_csv(path: str) -> pd.DataFrame:
    """Load a historical news dataset (needs at least 'date' and 'title' columns)."""
    df = pd.read_csv(path)
    missing = {"date", "title"} - set(df.columns)
    if missing:
        raise ValueError(f"News CSV is missing columns: {missing}")
    return df


def save_news_data(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "news_data.csv", index=False)
