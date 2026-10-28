import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.utils.config import LABELS, SENTIMENT_FEATURES

NAMES = {"majority_class": "Majority", "price_only": "Price-only", "price_sentiment": "Price+sentiment"}
ROWS = [
    ("Accuracy", "accuracy"),
    ("Macro F1", "macro_f1"),
    ("Weighted F1", "weighted_f1"),
    ("Directional accuracy", "directional_accuracy"),
    ("Directional calls (% days)", "call_rate"),
]


def predict_labels(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return bundle["classes"][bundle["model"].predict(df[bundle["features"]])]


def directional_accuracy(pred, next_return) -> tuple[float, int]:
    """Of the days where the model made an up/down call, how often was the sign right?

    Neutral predictions make no call and are excluded; days with exactly 0% return are
    excluded because they have no direction. Returns (accuracy, number_of_scored_calls).
    """
    pred = np.asarray(pred)
    actual = np.sign(np.asarray(next_return, dtype=float))
    mask = (pred != 1) & (actual != 0)
    if not mask.any():
        return float("nan"), 0
    pred_dir = np.where(pred == 2, 1.0, -1.0)
    return float((pred_dir[mask] == actual[mask]).mean()), int(mask.sum())


def score_predictions(y, pred, next_return) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    da, n_calls = directional_accuracy(pred, next_return)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "directional_accuracy": da,
        "directional_calls": n_calls,
        "call_rate": float((pred != 1).mean()),
    }


def _fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"


def _delta(a: float, b: float) -> str:
    return "n/a" if (a != a or b != b) else f"{a - b:+.3f}"


def compare_models(bundles: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[str, dict]:
    """Score every model (plus a majority-class baseline) on the SAME held-out future rows.

    bundles: {"price_only": bundle, "price_sentiment": bundle (optional)}
    """
    y = test_df["target"].to_numpy()
    nr = test_df["next_return"].to_numpy()
    majority = train_df["target"].mode()[0]

    results = {"majority_class": score_predictions(y, np.full(len(y), majority), nr)}
    preds = {}
    for key, bundle in bundles.items():
        preds[key] = predict_labels(bundle, test_df)
        results[key] = score_predictions(y, preds[key], nr)

    signs = np.sign(nr[nr != 0])
    always_up = float((signs > 0).mean()) if len(signs) else float("nan")

    cols = list(results)
    both = "price_only" in results and "price_sentiment" in results
    head = f"{'Metric':<28}" + "".join(f"{NAMES[c]:>17}" for c in cols) + (f"{'Δ (combined-price)':>21}" if both else "")
    lines = [
        f"Train rows: {len(train_df)} ({train_df['date'].min().date()} -> {train_df['date'].max().date()})",
        f"Test rows:  {len(test_df)} ({test_df['date'].min().date()} -> {test_df['date'].max().date()})",
        "", head, "-" * len(head),
    ]
    for label, key in ROWS:
        row = f"{label:<28}" + "".join(f"{_fmt(results[c][key]):>17}" for c in cols)
        if both:
            row += f"{_delta(results['price_sentiment'][key], results['price_only'][key]):>21}"
        lines.append(row)
    lines += [
        "",
        f"Always-bullish directional baseline (share of up days): {_fmt(always_up)}",
        "Directional accuracy only scores days where the model called up or down (not neutral).",
    ]

    if "price_sentiment" in bundles:
        b = bundles["price_sentiment"]
        imp = dict(zip(b["features"], b["model"].feature_importances_))
        share = sum(v for k, v in imp.items() if k in SENTIMENT_FEATURES)
        results["price_sentiment"]["sentiment_importance_share"] = float(share)
        lines.append(f"Share of model feature importance on sentiment features: {share:.1%}")

    labels_all = sorted(set(y.tolist()) | {int(c) for p in preds.values() for c in set(p.tolist())})
    for key, p in preds.items():
        lines += ["", f"Confusion matrix, {NAMES[key]} (rows=actual, cols=predicted; "
                  f"{', '.join(LABELS[int(l)] for l in labels_all)}):",
                  str(confusion_matrix(y, p, labels=labels_all))]
    if len(test_df) < 100:
        lines += ["", f"WARNING: only {len(test_df)} test rows. Differences of a few points are within noise."]
    results["_meta"] = {"train_rows": len(train_df), "test_rows": len(test_df), "always_up_directional": always_up}
    return "\n".join(lines), results
