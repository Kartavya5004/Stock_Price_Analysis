import json
import math

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.build_dataset import sentiment_coverage, sentiment_problems
from src.models.evaluate import compare_models
from src.utils.errors import InsufficientDataError
from src.utils.config import (
    FEATURES,
    MARKET_FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MIN_PRICE_ROWS,
    MODEL_PATH,
    PRICE_ONLY_MODEL_PATH,
    PROC_DIR,
    SPLIT_GAP,
    TEST_FRACTION,
)


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, gap: int = SPLIT_GAP):
    """Never shuffle time series: train on the past, test on the future.

    `gap` rows are dropped from the end of train: row i's label is the return of day i+1,
    so without it the last train label would be computed from the first test day's price move.
    """
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[: cut - gap], df.iloc[cut:]


def _fit(train_df: pd.DataFrame, features: list[str], name: str) -> dict:
    classes, y_train = np.unique(train_df["target"], return_inverse=True)  # XGB needs 0..k-1
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", n_jobs=-1,
    )
    model.fit(train_df[features], y_train)  # NaN sentiment (no news that day) is handled natively
    return {"model": model, "features": features, "classes": classes, "name": name}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def train(dataset: pd.DataFrame | None = None, allow_price_only: bool = False) -> dict:
    """Train a price-only model and (if the news data is dense enough) a price+sentiment model,
    then compare them on the same chronological hold-out."""
    if dataset is None:
        dataset = pd.read_csv(PROC_DIR / "final_dataset.csv", parse_dates=["date"])

    problems = sentiment_problems(dataset)
    if problems and not allow_price_only:
        cov = sentiment_coverage(dataset)
        raise ValueError(
            "Not enough news to train a price+sentiment model: " + "; ".join(problems) + ".\n"
            f"Dataset has {cov['rows']} rows, {cov['days_with_news']} with news.\n"
            "Rebuild with a historical news file (columns: date,title[,description]):\n"
            "    python main.py build AAPL --news-csv historical_news.csv\n"
            "or train only the price-only model with: python main.py train --allow-price-only"
        )

    if len(dataset) < MIN_PRICE_ROWS:
        raise InsufficientDataError(
            f"Only {len(dataset)} labelled trading days; need at least {MIN_PRICE_ROWS} to train and evaluate."
        )

    train_df, test_df = chronological_split(dataset)
    bundles = {"price_only": _fit(train_df, MARKET_FEATURES, "price-only")}
    if problems:
        print("WARNING: price+sentiment model skipped (" + "; ".join(problems) + ").")
    else:
        bundles["price_sentiment"] = _fit(train_df, FEATURES, "price+sentiment")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundles["price_only"], PRICE_ONLY_MODEL_PATH)
    joblib.dump(bundles.get("price_sentiment", bundles["price_only"]), MODEL_PATH)

    report, metrics = compare_models(bundles, train_df, test_df)
    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), indent=2))
    print(report)
    print(f"\nSaved models -> {MODEL_PATH}, {PRICE_ONLY_MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")
    return bundles
