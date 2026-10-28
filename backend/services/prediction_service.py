from src.models.predict import predict_ticker
from src.sentiment import sentiment_model
from src.sentiment.safety import is_injection
from src.utils.config import DISCLAIMER


def get_prediction(ticker: str) -> dict:
    return predict_ticker(ticker.upper())


def score_text(text: str) -> dict:
    if is_injection(text):  # untrusted text addressing the model: never scored, never echoed
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0,
                "flagged": True, "disclaimer": DISCLAIMER}
    row = sentiment_model.get_scorer().score([text]).iloc[0]
    out = {k: round(float(v), 4) for k, v in row.items()}
    out["score"] = max(-1.0, min(1.0, out["score"]))
    return {**out, "flagged": False, "disclaimer": DISCLAIMER}
