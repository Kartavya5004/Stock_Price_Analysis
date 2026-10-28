"""Unit tests for LLM Stock Price Predictor and Dummy Data Testing."""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from src.models.llm_predictor import LLMStockPredictor, StockPredictionOutput
from src.models.llm_time_series_model import TimeSeriesStockModel


def test_dummy_data_generation():
    predictor = LLMStockPredictor(provider="offline")
    prices_df, news_df = predictor.generate_dummy_data(
        days=30, ticker="AAPL", trend="bullish", seed=123
    )

    assert len(prices_df) == 30
    assert set(["date", "open", "high", "low", "close", "volume"]).issubset(prices_df.columns)
    assert not news_df.empty
    assert set(["date", "title", "description", "source"]).issubset(news_df.columns)
    assert (prices_df["high"] >= prices_df["low"]).all()
    assert (prices_df["close"] > 0).all()


def test_prepare_prompt():
    predictor = LLMStockPredictor(provider="offline")
    prices_df, news_df = predictor.generate_dummy_data(days=25, ticker="AAPL", seed=42)

    prompt = predictor.prepare_prompt(prices_df, news_df, ticker="AAPL", window_days=10)
    assert "AAPL" in prompt
    assert "RSI(14)" in prompt
    assert "MACD-Hist" in prompt
    assert "REQUIRED OUTPUT" in prompt


def test_offline_cot_prediction_schema():
    predictor = LLMStockPredictor(provider="offline")
    prices_df, news_df = predictor.generate_dummy_data(days=40, ticker="NVDA", trend="bullish", seed=42)

    pred = predictor.predict(prices_df, news_df, ticker="NVDA")

    assert isinstance(pred, StockPredictionOutput)
    assert pred.ticker == "NVDA"
    assert pred.current_price > 0
    assert pred.predicted_target_price > 0
    assert pred.predicted_direction in ["Bullish", "Bearish", "Neutral"]
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.reasoning) > 20
    assert len(pred.technical_summary) > 10
    assert len(pred.risk_factors) >= 1
    assert "Educational demonstration" in pred.disclaimer


def test_offline_cot_bearish_regime():
    predictor = LLMStockPredictor(provider="offline")
    prices_df, news_df = predictor.generate_dummy_data(days=50, ticker="TSLA", trend="bearish", seed=42)

    pred = predictor.predict(prices_df, news_df, ticker="TSLA")
    assert isinstance(pred, StockPredictionOutput)
    assert pred.predicted_direction in ["Bearish", "Neutral"]
    assert pred.predicted_target_price > 0


def test_evaluate_on_dummy_data():
    predictor = LLMStockPredictor(provider="offline")
    results = predictor.evaluate_on_dummy_data(days=35, window_size=15, ticker="MSFT", trend="bullish", seed=99)

    assert results["ticker"] == "MSFT"
    assert results["total_test_days"] == 20
    assert 0.0 <= results["directional_accuracy_pct"] <= 100.0
    assert results["mae"] >= 0.0
    assert results["rmse"] >= results["mae"] * 0.8  # Mathematical property of RMSE vs MAE
    assert len(results["predictions_log"]) == 20


def test_time_series_model():
    predictor = LLMStockPredictor(provider="offline")
    prices_df, news_df = predictor.generate_dummy_data(days=50, ticker="GOOGL", seed=77)

    ts_model = TimeSeriesStockModel()
    fit_metrics = ts_model.fit(prices_df, news_df)

    assert fit_metrics["samples"] > 10
    assert fit_metrics["train_return_mae"] >= 0.0

    prediction = ts_model.predict_next(prices_df, news_df)
    assert prediction["predicted_target_price"] > 0
    assert prediction["predicted_direction"] in ["Bullish", "Bearish", "Neutral"]
    assert "class_probabilities" in prediction


def test_fastapi_llm_routes():
    client = TestClient(app)

    # 1. Test GET /llm/dummy-test
    res = client.get("/llm/dummy-test?ticker=AAPL&trend=bullish&days=30")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "AAPL"
    assert "predicted_target_price" in data
    assert data["predicted_direction"] in ["Bullish", "Bearish", "Neutral"]

    # 2. Test GET /llm/dummy-backtest
    res_bt = client.get("/llm/dummy-backtest?ticker=AAPL&trend=bullish&days=35&window_size=15")
    assert res_bt.status_code == 200
    bt_data = res_bt.json()
    assert bt_data["ticker"] == "AAPL"
    assert "directional_accuracy_pct" in bt_data
    assert "mae" in bt_data
    assert len(bt_data["predictions_log"]) == 20

    # 3. Test POST /llm/predict
    res_post = client.post("/llm/predict", json={"ticker": "AMD", "trend": "volatile", "days": 35})
    assert res_post.status_code == 200
    post_data = res_post.json()
    assert post_data["ticker"] == "AMD"
    assert "predicted_target_price" in post_data
