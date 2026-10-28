import logging

from fastapi import APIRouter, HTTPException

from backend.schemas import SentimentOut, TextIn
from backend.services.prediction_service import score_text

router = APIRouter(prefix="/sentiment", tags=["sentiment"])
log = logging.getLogger(__name__)


@router.post("", response_model=SentimentOut)
def sentiment(body: TextIn):
    try:
        return score_text(body.text)
    except Exception:
        log.exception("sentiment model failed")
        raise HTTPException(status_code=503, detail="Sentiment model unavailable.")
