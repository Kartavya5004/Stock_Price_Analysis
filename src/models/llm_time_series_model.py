"""Time-Series Machine Learning / Neural Predictor for Stock Prices.

Learns sequential mapping from technical indicators + sentiment features to next-day
price levels and price movement directions.
"""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score

from src.features.technical_indicators import add_indicators
from src.models.llm_predictor import LLMStockPredictor


class TimeSeriesStockModel:
    """Quantitative Time-Series predictor that learns from technical and sentiment features."""

    def __init__(self, feature_columns: list[str] | None = None):
        self.feature_columns = feature_columns or [
            "return_1d",
            "return_5d",
            "close_sma10_ratio",
            "close_sma30_ratio",
            "ema_diff",
            "rsi_14",
            "macd_hist",
            "volume_ratio",
            "volatility_10",
        ]
        self.regressor = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
        )
        self.classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
        )
        self.is_fitted = False

    def prepare_dataset(
        self, prices_df: pd.DataFrame, news_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Enrich price data with indicators and next-day target price/return."""
        df = add_indicators(prices_df).copy()

        # Add next-day target price and return
        df["target_close"] = df["close"].shift(-1)
        df["target_return"] = (df["target_close"] - df["close"]) / df["close"]
        df["target_direction"] = np.where(
            df["target_return"] > 0.003,
            "Bullish",
            np.where(df["target_return"] < -0.003, "Bearish", "Neutral"),
        )

        # Drop initial NaN rows due to indicator rolling windows and the final row without target
        clean = df.dropna(subset=self.feature_columns + ["target_close"]).copy()
        return clean.reset_index(drop=True)

    def fit(self, prices_df: pd.DataFrame, news_df: pd.DataFrame | None = None) -> dict[str, float]:
        """Train the time-series model on historical or dummy stock data."""
        dataset = self.prepare_dataset(prices_df, news_df)
        if len(dataset) < 15:
            raise ValueError(f"Need at least 15 valid samples to fit model, got {len(dataset)}.")

        X = dataset[self.feature_columns]
        y_reg = dataset["target_return"]
        y_clf = dataset["target_direction"]

        self.regressor.fit(X, y_reg)
        self.classifier.fit(X, y_clf)
        self.is_fitted = True

        pred_ret = self.regressor.predict(X)
        pred_dir = self.classifier.predict(X)

        return {
            "samples": len(dataset),
            "train_return_mae": float(mean_absolute_error(y_reg, pred_ret)),
            "train_direction_accuracy": float(accuracy_score(y_clf, pred_dir)),
        }

    def predict_next(
        self, prices_df: pd.DataFrame, news_df: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Predict next session price and direction for the latest available bar."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_next().")

        feats = add_indicators(prices_df)
        latest = feats.dropna(subset=self.feature_columns).iloc[[-1]]

        last_close = float(latest["close"].iloc[0])
        X = latest[self.feature_columns]

        pred_return = float(self.regressor.predict(X)[0])
        pred_dir = str(self.classifier.predict(X)[0])
        pred_probs = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, pred_probs)}
        target_price = round(last_close * (1 + pred_return), 2)

        return {
            "current_price": last_close,
            "predicted_target_price": target_price,
            "predicted_direction": pred_dir,
            "predicted_return_percent": round(pred_return * 100, 2),
            "class_probabilities": prob_dict,
        }
