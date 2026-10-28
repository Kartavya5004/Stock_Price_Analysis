"""Endpoints and static files the dashboard depends on: /history, /metrics, and the frontend mount."""
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import FRONTEND_DIR, app
from backend.routes import metrics as metrics_route
from backend.services import history_service
from src.utils.config import DISCLAIMER
from src.utils.errors import NoDataError
from tests.conftest import make_prices


@pytest.fixture
def client():
    return TestClient(app)


def _serve_prices(monkeypatch, prices):
    monkeypatch.setattr(history_service, "fetch_stock_data", lambda t, start=None, end=None: prices.copy())


# =============== /history ===============
def test_history_returns_requested_bars_with_moving_averages(client, monkeypatch):
    prices = make_prices(400)
    _serve_prices(monkeypatch, prices)
    r = client.get("/history/aapl?days=100")  # lower-case input is accepted
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert len(body["points"]) == 100
    first, last = body["points"][0], body["points"][-1]
    assert first["sma30"] is not None  # warm-up bars are fetched but not returned
    assert last["date"] == str(prices["date"].max().date())
    assert last["close"] == pytest.approx(prices["close"].iloc[-1], abs=0.01)
    assert last["sma10"] == pytest.approx(prices["close"].tail(10).mean(), abs=0.01)
    assert body["disclaimer"] == DISCLAIMER


def test_history_unknown_ticker_is_404(client, monkeypatch):
    def boom(*a, **k):
        raise NoDataError("No price data returned for 'ZZZZ'")
    monkeypatch.setattr(history_service, "fetch_stock_data", boom)
    assert client.get("/history/ZZZZ").status_code == 404


def test_history_single_bar_is_422(client, monkeypatch):
    _serve_prices(monkeypatch, make_prices(400).head(1))
    assert client.get("/history/NEWIPO").status_code == 422


def test_history_provider_failure_is_502_and_hides_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret upstream stack trace")
    monkeypatch.setattr(history_service, "fetch_stock_data", boom)
    r = client.get("/history/AAPL")
    assert r.status_code == 502 and "secret" not in r.text


@pytest.mark.parametrize("path", ["/history/-AAPL", "/history/AAPL?days=5", "/history/AAPL?days=5000"])
def test_history_rejects_bad_input(client, path):
    assert client.get(path).status_code == 422


# =============== /metrics ===============
SAMPLE_METRICS = {
    "majority_class": {"accuracy": 0.41, "macro_f1": 0.19, "weighted_f1": 0.24,
                       "directional_accuracy": None, "directional_calls": 0, "call_rate": 0.0},
    "price_only": {"accuracy": 0.43, "macro_f1": 0.36, "weighted_f1": 0.40,
                   "directional_accuracy": 0.52, "directional_calls": 118, "call_rate": 0.51},
    "_meta": {"train_rows": 920, "test_rows": 230, "always_up_directional": 0.53},
}


def test_metrics_503_when_not_trained(client, monkeypatch, tmp_path):
    monkeypatch.setattr(metrics_route, "METRICS_PATH", tmp_path / "missing.json")
    r = client.get("/metrics")
    assert r.status_code == 503 and "main.py train" in r.json()["detail"]


def test_metrics_returns_models_meta_and_disclaimer(client, monkeypatch, tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(SAMPLE_METRICS))
    monkeypatch.setattr(metrics_route, "METRICS_PATH", path)
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["models"]) == {"majority_class", "price_only"}
    assert body["models"]["majority_class"]["directional_accuracy"] is None  # NaN survives as null
    assert body["meta"]["test_rows"] == 230
    assert body["disclaimer"] == DISCLAIMER


def test_metrics_unreadable_file_is_503(client, monkeypatch, tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json")
    monkeypatch.setattr(metrics_route, "METRICS_PATH", path)
    assert client.get("/metrics").status_code == 503


# =============== dashboard files ===============
def test_dashboard_is_served_and_api_routes_still_win(client):
    page = client.get("/")
    assert page.status_code == 200 and "Next-day signal" in page.text
    assert client.get("/js/main.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok"  # not shadowed by the static mount


def test_dashboard_shows_the_api_disclaimer_without_waiting_for_the_network():
    """README: the dashboard must show the same disclaimer. The page ships it as built-in text."""
    html = (FRONTEND_DIR / "index.html").read_text()
    assert DISCLAIMER in html
