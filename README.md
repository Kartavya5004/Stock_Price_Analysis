# Stock_Price_Analysis

Next-day direction prediction (bullish / neutral / bearish) from technical indicators + FinBERT news sentiment.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add NEWS_API_KEY (optional)
```

## Run
```bash
python test_llm_model.py              # test LLM stock prediction on dummy data (quickest demo)
python main.py llm-predict AAPL       # LLM prediction on synthetic dummy data
python main.py llm-predict AAPL --eval # LLM rolling backtest on dummy data
python main.py build AAPL            # or: --news-csv historical_news.csv
python main.py train                 # price-only vs price+sentiment on the same hold-out
python main.py predict AAPL
python main.py serve                 # dashboard: http://127.0.0.1:8000/   API docs: /docs
```

## How news is used
- Predictions are assumed to be made at the US market close. Articles are converted to US/Eastern; anything published at/after 16:00 ET counts for the next trading day. Weekend/holiday news maps to the next session.
- A day with no articles has **missing** sentiment (NaN), not neutral 0. `news_count` is 0 on those days.
- With news available, the dataset is limited to the first-to-last news date, so the model never trains on periods where sentiment was unobserved.
- `train` refuses to build the price+sentiment model unless the window has >= 200 trading days and >= 30% of them have news (see `config.py`). NewsAPI's free tier (~28 days) will not pass; use `--news-csv`. `python main.py train --allow-price-only` trains only the baseline.

## Evaluation
`train` prints, on one chronological hold-out (with a 1-row gap at the boundary): accuracy, macro/weighted F1 and directional accuracy for a majority baseline, the price-only model and the price+sentiment model, plus confusion matrices. Results are also saved to `models/metrics.json`. Directional accuracy only scores days where the model called up or down. With a few hundred test rows, differences of a few points are noise.

## Dashboard
`frontend/` is plain HTML, CSS and ES modules: no Node, no build step, no CDN. `python main.py serve` serves it at `/`, so the page and the API share one origin.

- **Signal**: Bearish / Neutral / Bullish probabilities drawn as one proportional bar, last close, which model produced the call, and the 3-day news sentiment (or why there is none).
- **Price history**: close plus 10- and 30-day averages (3M / 6M / 1Y), with a hover or touch readout.
- **Evaluation**: `models/metrics.json` as a table (majority baseline vs price-only vs price+sentiment), with the always-up baseline and a sample-size caveat. The column used for the current signal is marked.
- **Check a headline**: scores any text through `POST /sentiment`; instruction-like text is shown as "Not scored".
- Errors map to plain advice: 404 unknown ticker, 422 too little history or bad input, 503 no trained model (or no evaluation yet), 502 provider failure, and "can't reach the API".

To develop the page against a separate server, open it with `?api=http://127.0.0.1:8000` (CORS already allows `localhost:3000` and `localhost:5173`). Opening `index.html` straight from disk will not work, because browsers block ES modules on `file://`.

## Endpoints
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/predict/{ticker}` | Next-day signal, probabilities, recent sentiment |
| GET | `/history/{ticker}?days=252` | Last N trading bars with 10/30-day averages (N: 20-1000) |
| GET | `/metrics` | Hold-out evaluation from `models/metrics.json` (503 until `train` has run) |
| POST | `/sentiment` | FinBERT score for one text (1-1000 chars) |
| GET | `/health` | Liveness plus the disclaimer |

## API safety
- Every prediction/sentiment response and `/health` carries a `disclaimer` field, and every response has an `X-Disclaimer` header.
- `GET /predict/{ticker}`: ticker must match `^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$` (else 422). Unknown ticker 404, too little history 422, no trained model 503, provider failure 502 (details logged, not returned).
- `POST /sentiment`: text must be 1-1000 chars and non-blank (else 422).
- Headlines containing instruction-like text ("ignore previous instructions", "SYSTEM:", ...) are never scored: excluded from daily sentiment, and `/sentiment` returns `flagged: true`. This is a keyword heuristic, not a guarantee. Scoring is by FinBERT, a classifier that cannot follow instructions; if you swap in an LLM scorer, also delimit the headline, require strict JSON and clamp the score.
- If the news provider is down, `/predict` still answers using price features only (sentiment is reported as null).
- The dashboard shows the same disclaimer in a permanent band under the header and again in the footer (text comes from `/health`; a built-in copy is shown if the API is unreachable).

## Tests
```bash
python -m pytest tests            # offline: Yahoo/NewsAPI mocked, FinBERT replaced by a keyword stub
RUN_FINBERT=1 python -m pytest tests -k real_finbert   # opt-in: downloads FinBERT
```
`tests/test_dashboard_api.py` covers `/history`, `/metrics` and the static mount (including that the page ships the same disclaimer as `config.py`).
`tests/test_spec_cases.py` covers the five spec cases: valid ticker, missing/short data, strongly positive/negative news, look-ahead leakage (including a canary that proves the check catches a deliberate leak), and prompt injection.

## Notes
- The split is chronological (no shuffling) to avoid lookahead leakage.
- NewsAPI's free tier only covers ~30 days; for real training use a historical news CSV with `date,title[,description]` columns.
- Not financial advice.
