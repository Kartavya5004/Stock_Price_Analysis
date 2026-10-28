import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "stock_model.pkl"  # best available model (price+sentiment if trainable)
PRICE_ONLY_MODEL_PATH = MODEL_DIR / "stock_model_price_only.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
DEFAULT_TICKER = "AAPL"
START_DATE = "2020-01-01"

FINBERT_MODEL = "ProsusAI/finbert"

# Next-day return beyond +/- this threshold => bullish / bearish, else neutral
NEUTRAL_THRESHOLD = 0.005
TEST_FRACTION = 0.2  # last 20% of dates held out (chronological split)
SPLIT_GAP = 1        # rows dropped at the train/test boundary (the last train label spans the first test day)

# News timing: predictions are assumed to be made at the US market close.
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16  # articles published at/after this hour (ET) count for the NEXT trading day

# The price+sentiment model is only trained when the news window is dense enough to learn from.
MIN_SENTIMENT_ROWS = 200   # trading days inside the news window (~10 months)
MIN_NEWS_COVERAGE = 0.30   # share of those days with at least one article

MARKET_FEATURES = [
    "return_1d", "return_5d", "close_sma10_ratio", "close_sma30_ratio",
    "ema_diff", "rsi_14", "macd", "macd_signal", "macd_hist",
    "volume_change", "volume_ratio", "volatility_10",
]
SENTIMENT_FEATURES = ["sent_mean", "sent_pos_mean", "sent_neg_mean", "news_count", "sent_mean_3d"]
FEATURES = MARKET_FEATURES + SENTIMENT_FEATURES

LABELS = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

# ---- API safety boundary ----
DISCLAIMER = (
    "Educational demonstration only. Not financial or investment advice. "
    "Predictions are uncertain and can be wrong. No live trading."
)
TICKER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"
MAX_TEXT_CHARS = 1000   # longest text accepted by POST /sentiment
MIN_PRICE_ROWS = 120    # fewest labelled trading days we will train on
