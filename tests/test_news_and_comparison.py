import numpy as np
import pandas as pd
import pytest

from src.features.build_dataset import (
    add_sentiment_features,
    align_to_trading_days,
    build_final_dataset,
    sentiment_problems,
)
from src.features.technical_indicators import add_indicators
from src.models import train as train_mod
from src.models.evaluate import directional_accuracy, score_predictions
from src.preprocessing.clean_news import clean_news
from src.preprocessing.clean_stock_data import clean_stock_data


def _news(dates):
    return pd.DataFrame({"date": dates, "title": [f"Headline number {i} about the company" for i in range(len(dates))]})


# ---------- news timing ----------
def test_clean_news_converts_to_eastern_and_shifts_after_close():
    out = clean_news(_news([
        "2024-01-05T15:00:00Z",   # 10:00 ET -> same day
        "2024-01-05T22:30:00Z",   # 17:30 ET -> after the close -> next day
        "2024-01-05",             # date-only -> as given
        "2024-01-05 20:00:00",    # naive, 20:00 wall-clock -> after close -> next day
    ])).sort_values("text")
    assert list(out["date"]) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06"),
                                 pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06")]


def test_weekend_news_maps_to_next_trading_day():
    cal = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"]))
    news = clean_news(_news(["2024-01-06", "2024-01-07", "2024-01-05"]))
    out = align_to_trading_days(news, cal)
    assert sorted(out["date"]) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-08")]


def test_news_after_last_bar_dropped_in_training_kept_for_live():
    cal = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-05"]))
    news = clean_news(_news(["2024-01-07"]))
    assert align_to_trading_days(news, cal).empty
    live = align_to_trading_days(news, cal, allow_after_last=True)
    assert list(live["date"]) == [pd.Timestamp("2024-01-05")]


# ---------- missing vs neutral ----------
def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n), "close": close,
        "volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    })


def _daily(dates, rng):
    return pd.DataFrame({
        "date": dates, "sent_mean": rng.normal(0, 0.3, len(dates)),
        "sent_pos_mean": rng.uniform(0.2, 0.6, len(dates)), "sent_neg_mean": rng.uniform(0.1, 0.4, len(dates)),
        "news_count": rng.integers(1, 6, len(dates)).astype(float),
    })


def test_no_news_day_is_nan_not_zero():
    feats = add_indicators(clean_stock_data(_prices()))
    daily = _daily(feats["date"].iloc[100:110], np.random.default_rng(1))
    df = add_sentiment_features(feats, daily)
    assert df.loc[:99, "sent_mean"].isna().all()
    assert (df.loc[:99, "news_count"] == 0).all()
    assert df.loc[100:109, "sent_mean"].notna().all()


def test_dataset_restricted_to_news_window():
    feats = add_indicators(clean_stock_data(_prices()))
    dates = feats["date"].iloc[150:350]
    final = build_final_dataset(feats, _daily(dates, np.random.default_rng(2)))
    assert final["date"].min() >= dates.min() and final["date"].max() <= dates.max()
    assert len(final) > 150


def test_no_news_keeps_full_history():
    feats = add_indicators(clean_stock_data(_prices()))
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "sent_mean": [], "sent_pos_mean": [],
                          "sent_neg_mean": [], "news_count": []})
    final = build_final_dataset(feats, empty)
    assert len(final) > 300 and final["sent_mean"].isna().all()


def test_sparse_news_is_flagged():
    feats = add_indicators(clean_stock_data(_prices()))
    thin = _daily(feats["date"].iloc[-20:], np.random.default_rng(3))  # ~4 weeks, like NewsAPI free tier
    final = build_final_dataset(feats, thin)
    assert sentiment_problems(final)


# ---------- metrics ----------
def test_directional_accuracy_hand_computed():
    pred = np.array([2, 2, 0, 1, 0, 2])
    ret = np.array([0.01, -0.02, -0.01, 0.03, 0.02, 0.0])
    # scored calls: (up,+) ok, (up,-) wrong, (down,-) ok, (down,+) wrong; neutral and 0% return excluded
    acc, n = directional_accuracy(pred, ret)
    assert (acc, n) == (0.5, 4)


def test_directional_accuracy_no_calls_is_nan():
    acc, n = directional_accuracy(np.array([1, 1]), np.array([0.01, -0.01]))
    assert np.isnan(acc) and n == 0


def test_score_predictions_perfect():
    y = np.array([0, 1, 2, 2])
    m = score_predictions(y, y, np.array([-0.01, 0.0, 0.02, 0.01]))
    assert m["accuracy"] == m["macro_f1"] == m["directional_accuracy"] == 1.0


# ---------- split + end-to-end ----------
def test_split_has_gap_and_no_overlap():
    df = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=100), "target": 1})
    tr, te = train_mod.chronological_split(df, 0.2, gap=1)
    assert tr["date"].max() < te["date"].min()
    assert len(df) - len(tr) - len(te) == 1


def test_train_compares_both_models(tmp_path, monkeypatch):
    for name, val in {"MODEL_DIR": tmp_path, "MODEL_PATH": tmp_path / "m.pkl",
                      "PRICE_ONLY_MODEL_PATH": tmp_path / "p.pkl", "METRICS_PATH": tmp_path / "metrics.json"}.items():
        monkeypatch.setattr(train_mod, name, val)
    rng = np.random.default_rng(4)
    feats = add_indicators(clean_stock_data(_prices(n=500)))
    mask_days = feats["date"].sample(frac=0.7, random_state=1).sort_values()  # ~70% of days have news
    final = build_final_dataset(feats, _daily(mask_days, rng))
    bundles = train_mod.train(final)
    assert set(bundles) == {"price_only", "price_sentiment"}
    assert not set(bundles["price_only"]["features"]) & {"sent_mean", "news_count"}
    import json
    m = json.loads((tmp_path / "metrics.json").read_text())
    for k in ("majority_class", "price_only", "price_sentiment"):
        assert {"accuracy", "macro_f1", "weighted_f1", "directional_accuracy"} <= set(m[k])


def test_train_refuses_sparse_news_unless_allowed(tmp_path, monkeypatch):
    for name, val in {"MODEL_DIR": tmp_path, "MODEL_PATH": tmp_path / "m.pkl",
                      "PRICE_ONLY_MODEL_PATH": tmp_path / "p.pkl", "METRICS_PATH": tmp_path / "metrics.json"}.items():
        monkeypatch.setattr(train_mod, name, val)
    feats = add_indicators(clean_stock_data(_prices(n=500)))
    final = build_final_dataset(feats, _daily(feats["date"].iloc[-20:], np.random.default_rng(5)))
    with pytest.raises(ValueError, match="Not enough news"):
        train_mod.train(final)
