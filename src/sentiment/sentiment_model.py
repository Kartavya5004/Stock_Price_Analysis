from functools import lru_cache

import pandas as pd

from src.sentiment.safety import split_flagged
from src.utils.config import FINBERT_MODEL


class FinBertSentiment:
    """Thin wrapper over the FinBERT text-classification pipeline (CPU-friendly)."""

    def __init__(self, model_name: str = FINBERT_MODEL, device: int = -1):
        from transformers import pipeline  # lazy import: heavy

        self.pipe = pipeline(
            "text-classification", model=model_name, top_k=None,
            truncation=True, max_length=128, device=device,
        )

    def score(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        """Return positive/negative/neutral probabilities and score = pos - neg."""
        results = self.pipe(list(texts), batch_size=batch_size)
        rows = [{d["label"].lower(): d["score"] for d in r} for r in results]
        out = pd.DataFrame(rows)[["positive", "negative", "neutral"]]
        out["score"] = out["positive"] - out["negative"]
        return out


@lru_cache(maxsize=1)
def get_scorer() -> FinBertSentiment:
    return FinBertSentiment()


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    """Attach FinBERT scores to a cleaned news frame.

    Headlines that contain instruction-like text are dropped first (see safety.py).
    """
    news, _ = split_flagged(news)
    if news.empty:
        return news.assign(positive=pd.Series(dtype="float64"), negative=pd.Series(dtype="float64"),
                           neutral=pd.Series(dtype="float64"), score=pd.Series(dtype="float64"))
    scores = get_scorer().score(news["text"].tolist())
    return pd.concat([news.reset_index(drop=True), scores], axis=1)


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse article-level scores to one row per day."""
    if scored.empty:
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        for col in ["sent_mean", "sent_pos_mean", "sent_neg_mean", "news_count"]:
            empty[col] = pd.Series(dtype="float64")
        return empty
    g = scored.groupby("date")
    return pd.DataFrame({
        "sent_mean": g["score"].mean(),
        "sent_pos_mean": g["positive"].mean(),
        "sent_neg_mean": g["negative"].mean(),
        "news_count": g["score"].size(),
    }).reset_index()


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """cleaned news -> daily sentiment features (empty-safe)."""
    if news.empty:
        return aggregate_daily(news)
    return aggregate_daily(score_news(news))
