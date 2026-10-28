from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import history, metrics, prediction, sentiment, llm_prediction
from src.utils.config import DISCLAIMER, ROOT

FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="Stock Prediction API",
    description=DISCLAIMER,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(prediction.router)
app.include_router(sentiment.router)
app.include_router(history.router)
app.include_router(metrics.router)
app.include_router(llm_prediction.router)


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.middleware("http")
async def add_disclaimer_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Disclaimer"] = "Educational only; not financial advice"
    return response


# Dashboard (no build step). Mounted last so every API route above takes precedence.
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
