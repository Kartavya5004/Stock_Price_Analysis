import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from src.features.build_dataset import align_to_trading_days, build_final_dataset
from src.features.technical_indicators import add_indicators
from src.models import predict as predict_mod
from src.models import train as train_mod
from src.preprocessing.clean_news import clean_news
from src.preprocessing.clean_stock_data import clean_stock_data
from src.sentiment import sentiment_model
from src.sentiment.sentiment_model import aggregate_daily, score_news

POS = ("surges", "beats", "record", "soars", "upgrade", "profit")
NEG = ("plunges", "misses", "fraud", "bankruptcy", "downgrade", "lawsuit")


class FakeScorer:
    """Keyword stand-in for FinBERT (not available offline). Records every text it is asked to score."""

    def __init__(self):
        self.seen: list[str] = []

    def score(self, texts, batch_size=16):
        rows = []
        for t in texts:
            self.seen.append(t)
            p = sum(w in t.lower() for w in POS)
            n = sum(w in t.lower() for w in NEG)
            pos, neg = (0.9, 0.05) if p > n else (0.05, 0.9) if n > p else (0.1, 0.1)
            rows.append({"positive": pos, "negative": neg, "neutral": 1 - pos - neg})
        out = pd.DataFrame(rows)
        out["score"] = out["positive"] - out["negative"]
        return out


def make_prices(n=700, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n), "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": rng.integers(1_000_000, 2_000_000, n).astype(float), "ticker": "TEST",
    })


def make_news(prices: pd.DataFrame, frac=0.8, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(prices["date"]):
        if rng.random() < frac:
            word = rng.choice(POS + NEG)
            rows.append({"date": f"{d.date()}T15:00:00Z", "title": f"Company {word} in session number {i}", "description": None})
    return pd.DataFrame(rows)


def build_dataset(prices, news_raw):
    feats = add_indicators(clean_stock_data(prices))
    news = align_to_trading_days(clean_news(news_raw), feats["date"])
    return build_final_dataset(feats, aggregate_daily(score_news(news)))


@pytest.fixture
def scorer(monkeypatch):
    s = FakeScorer()
    monkeypatch.setattr(sentiment_model, "get_scorer", lambda: s)
    return s


@pytest.fixture
def trained(monkeypatch, tmp_path, scorer):
    """A real (small) price+sentiment model trained on synthetic data, with all paths in tmp."""
    for name, val in {"MODEL_DIR": tmp_path, "MODEL_PATH": tmp_path / "m.pkl",
                      "PRICE_ONLY_MODEL_PATH": tmp_path / "p.pkl", "METRICS_PATH": tmp_path / "x.json"}.items():
        monkeypatch.setattr(train_mod, name, val)
    prices = make_prices()
    bundles = train_mod.train(build_dataset(prices, make_news(prices)))
    monkeypatch.setattr(predict_mod, "_bundle", bundles["price_sentiment"])
    scorer.seen.clear()  # forget the headlines scored during training; tests inspect only what they trigger
    return prices, bundles


@pytest.fixture
def api(monkeypatch, trained):
    prices, _ = trained
    last = prices["date"].max()
    recent = pd.DataFrame({
        "date": [f"{(last - pd.Timedelta(days=k)).date()}T15:00:00Z" for k in range(5)],
        "title": [f"Company {w} again number {k}" for k, w in enumerate(["soars", "beats", "misses", "record", "profit"])],
        "description": None,
    })
    monkeypatch.setattr(predict_mod, "fetch_stock_data", lambda t, start=None, end=None: prices.copy())
    monkeypatch.setattr(predict_mod, "fetch_news", lambda t, days=7: recent)
    return TestClient(app), prices
