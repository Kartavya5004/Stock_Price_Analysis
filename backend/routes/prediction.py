import logging

from fastapi import APIRouter, HTTPException, Path

from backend.schemas import PredictionOut
from backend.services.prediction_service import get_prediction
from src.utils.config import TICKER_PATTERN
from src.utils.errors import InsufficientDataError, NoDataError

router = APIRouter(prefix="/predict", tags=["prediction"])
log = logging.getLogger(__name__)


@router.get("/{ticker}", response_model=PredictionOut)
def predict(ticker: str = Path(..., pattern=TICKER_PATTERN, max_length=10, description="e.g. AAPL, BRK-B")):
    try:
        return get_prediction(ticker)
    except NoDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # upstream provider failure etc.: log details, don't leak them
        log.exception("prediction failed for %s", ticker)
        raise HTTPException(status_code=502, detail="Upstream data provider error. Try again later.")
