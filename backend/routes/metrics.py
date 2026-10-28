import json
import logging

from fastapi import APIRouter, HTTPException

from backend.schemas import MetricsOut
from src.utils.config import METRICS_PATH

router = APIRouter(prefix="/metrics", tags=["evaluation"])
log = logging.getLogger(__name__)


@router.get("", response_model=MetricsOut)
def metrics():
    """Hold-out evaluation written by `python main.py train` (models/metrics.json)."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="No evaluation found. Run: python main.py train")
    try:
        raw = json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        log.exception("could not read %s", METRICS_PATH)
        raise HTTPException(status_code=503, detail="Evaluation file is unreadable. Re-run: python main.py train")
    meta = raw.pop("_meta", {})
    return {"models": raw, "meta": meta}
