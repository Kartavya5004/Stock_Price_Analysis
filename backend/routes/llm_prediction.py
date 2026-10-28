"""FastAPI routes for LLM Stock Price Prediction and Dummy Data Testing."""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.llm_predictor import LLMStockPredictor, StockPredictionOutput
from src.utils.config import DISCLAIMER

router = APIRouter(prefix="/llm", tags=["LLM Stock Prediction"])

_predictor = LLMStockPredictor(provider="auto")


class DummyBacktestResponse(BaseModel):
    ticker: str
    trend_regime: str
    total_test_days: int
    directional_accuracy_pct: float
    mae: float
    rmse: float
    mape_pct: float
    predictions_log: list[dict]
    disclaimer: str = DISCLAIMER


class PredictCustomRequest(BaseModel):
    ticker: str = "AAPL"
    trend: Literal["bullish", "bearish", "neutral", "volatile"] = "bullish"
    days: int = Field(default=60, ge=15, le=365)


@router.get("/dummy-test", response_model=StockPredictionOutput)
def test_dummy_prediction(
    ticker: str = Query("AAPL", description="Stock ticker symbol"),
    trend: Literal["bullish", "bearish", "neutral", "volatile"] = Query(
        "bullish", description="Market regime to simulate"
    ),
    days: int = Query(60, ge=20, le=200, description="Number of synthetic days"),
):
    """Generate dummy stock prices & financial news, then run LLM stock prediction."""
    try:
        prices_df, news_df = _predictor.generate_dummy_data(
            days=days, ticker=ticker.upper(), trend=trend
        )
        return _predictor.predict(prices_df, news_df, ticker=ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM prediction failed: {str(e)}")


@router.get("/dummy-backtest", response_model=DummyBacktestResponse)
def test_dummy_backtest(
    ticker: str = Query("AAPL", description="Stock ticker symbol"),
    trend: Literal["bullish", "bearish", "neutral", "volatile"] = Query(
        "bullish", description="Market regime to simulate"
    ),
    days: int = Query(45, ge=25, le=120, description="Total days to simulate"),
    window_size: int = Query(15, ge=10, le=30, description="Rolling historical window size"),
):
    """Run a multi-day rolling backtest on dummy data and return accuracy & error metrics."""
    try:
        eval_result = _predictor.evaluate_on_dummy_data(
            days=days, window_size=window_size, ticker=ticker.upper(), trend=trend
        )
        return DummyBacktestResponse(**eval_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.post("/predict", response_model=StockPredictionOutput)
def predict_stock_llm(req: PredictCustomRequest):
    """Run LLM stock prediction with specified parameters."""
    prices_df, news_df = _predictor.generate_dummy_data(
        days=req.days, ticker=req.ticker.upper(), trend=req.trend
    )
    return _predictor.predict(prices_df, news_df, ticker=req.ticker.upper())
