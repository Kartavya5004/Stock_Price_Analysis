import logging

from fastapi import APIRouter, HTTPException, Path, Query

from backend.schemas import HistoryOut
from backend.services.history_service import get_history
from src.utils.config import TICKER_PATTERN
from src.utils.errors import InsufficientDataError, NoDataError

router = APIRouter(prefix="/history", tags=["history"])
log = logging.getLogger(__name__)


@router.get("/{ticker}", response_model=HistoryOut)
def history(
    ticker: str = Path(..., pattern=TICKER_PATTERN, max_length=10, description="e.g. AAPL, BRK-B"),
    days: int = Query(252, ge=20, le=1000, description="Trading days to return"),
):
    try:
        return get_history(ticker, days)
    except NoDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InsufficientDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:  # upstream provider failure etc.: log details, don't leak them
        log.exception("history failed for %s", ticker)
        raise HTTPException(status_code=502, detail="Upstream data provider error. Try again later.")
