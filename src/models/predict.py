from datetime import date, timedelta

import warnings

import joblib
import pandas as pd
import requests

from src.data.news_fetcher import fetch_news
from src.data.stock_fetcher import fetch_stock_data
from src.features.build_dataset import add_sentiment_features, align_to_trading_days
from src.features.technical_indicators import add_indicators
from src.preprocessing.clean_news import clean_news
from src.preprocessing.clean_stock_data import clean_stock_data
from src.sentiment.sentiment_model import daily_sentiment
from src.utils.config import DISCLAIMER, LABELS, MARKET_FEATURES, MODEL_PATH, SENTIMENT_FEATURES
from src.utils.errors import InsufficientDataError

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run: python main.py train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_ticker(ticker: str) -> dict:
    bundle = load_bundle()
    model, features, classes = bundle["model"], bundle["features"], bundle["classes"]
    uses_sentiment = any(f in SENTIMENT_FEATURES for f in features)

    start = (date.today() - timedelta(days=220)).isoformat()
    prices = clean_stock_data(fetch_stock_data(ticker, start=start))
    if uses_sentiment:
        # news after the last bar (e.g. over the weekend) belongs to the latest known bar
        try:
            raw_news = fetch_news(ticker, days=7)
        except requests.RequestException as e:  # news is optional input: degrade to missing sentiment
            warnings.warn(f"News fetch failed ({type(e).__name__}); predicting without sentiment.")
            raw_news = pd.DataFrame(columns=["date", "title", "description", "source"])
        news = align_to_trading_days(clean_news(raw_news), prices["date"], allow_after_last=True)
        daily = daily_sentiment(news)
    else:
        daily = daily_sentiment(pd.DataFrame())
    df = add_sentiment_features(add_indicators(prices), daily)

    # Only price features must be present; missing sentiment (no recent news) is valid input.
    valid = df.dropna(subset=[f for f in features if f in MARKET_FEATURES])
    if valid.empty:
        raise InsufficientDataError(
            f"Only {len(prices)} price rows for '{ticker}'; not enough history to compute indicators."
        )
    row = valid.iloc[[-1]]
    proba = model.predict_proba(row[features])[0]
    probs = {LABELS[int(c)]: round(float(p), 4) for c, p in zip(classes, proba)}
    best = max(probs, key=probs.get)
    recent = row["sent_mean_3d"].iloc[0]
    return {
        "ticker": ticker,
        "model": bundle.get("name", "unknown"),
        "as_of": str(row["date"].iloc[0].date()),
        "last_close": round(float(row["close"].iloc[0]), 2),
        "signal": best,
        "confidence": probs[best],
        "probabilities": probs,
        "recent_sentiment": None if pd.isna(recent) else round(float(recent), 4),
        "disclaimer": DISCLAIMER,
    }
