from pydantic import BaseModel, Field, field_validator

from src.utils.config import DISCLAIMER, MAX_TEXT_CHARS


class PredictionOut(BaseModel):
    ticker: str
    model: str
    as_of: str
    last_close: float
    signal: str = Field(description="Bearish | Neutral | Bullish (next trading day)")
    confidence: float = Field(ge=0, le=1, description="Raw model probability, not calibrated")
    probabilities: dict[str, float]
    recent_sentiment: float | None = Field(None, description="3-day mean sentiment; null if no recent news")
    disclaimer: str = DISCLAIMER


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class SentimentOut(BaseModel):
    positive: float
    negative: float
    neutral: float
    score: float = Field(ge=-1, le=1)
    flagged: bool = Field(False, description="True if the text looked like an instruction to the model and was not scored")
    disclaimer: str = DISCLAIMER


class HistoryPoint(BaseModel):
    date: str
    close: float
    sma10: float | None = Field(None, description="10-day simple moving average; null until enough history")
    sma30: float | None = Field(None, description="30-day simple moving average; null until enough history")


class HistoryOut(BaseModel):
    ticker: str
    points: list[HistoryPoint]
    disclaimer: str = DISCLAIMER


class MetricsOut(BaseModel):
    models: dict[str, dict[str, float | int | None]] = Field(
        description="Hold-out scores keyed by majority_class / price_only / price_sentiment"
    )
    meta: dict[str, float | int | None] = Field(description="Train/test sizes and the always-up directional baseline")
    disclaimer: str = DISCLAIMER
