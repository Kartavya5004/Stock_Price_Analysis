"""CLI entry point.

  python main.py build   AAPL [--news-csv path.csv]   # fetch -> features -> final_dataset.csv
  python main.py train [--allow-price-only]           # price-only vs price+sentiment, same hold-out
  python main.py predict AAPL                         # live prediction
  python main.py llm-predict AAPL [--dummy] [--eval]  # LLM stock price prediction with dummy test
  python main.py serve                                # FastAPI on :8000
"""
import argparse

import pandas as pd

from src.utils.config import DEFAULT_TICKER, PROC_DIR


def build(ticker: str, news_csv: str | None = None) -> pd.DataFrame:
    from src.data.news_fetcher import fetch_news, load_news_csv, save_news_data
    from src.data.stock_fetcher import fetch_stock_data, save_stock_data
    from src.features.build_dataset import (
        align_to_trading_days,
        build_final_dataset,
        sentiment_coverage,
    )
    from src.features.technical_indicators import add_indicators
    from src.preprocessing.clean_news import clean_news
    from src.preprocessing.clean_stock_data import clean_stock_data
    from src.sentiment.sentiment_model import aggregate_daily, score_news

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    stock = fetch_stock_data(ticker)
    save_stock_data(stock)
    feats = add_indicators(clean_stock_data(stock))
    feats.to_csv(PROC_DIR / "stock_features.csv", index=False)

    raw_news = load_news_csv(news_csv) if news_csv else fetch_news(ticker)
    save_news_data(raw_news)
    # clean_news() puts each article on its market date; align maps weekends/holidays to the next session
    news = align_to_trading_days(clean_news(raw_news), feats["date"])
    if news.empty:
        print("No usable news: building a price-only dataset (sentiment will be missing).")
        daily = aggregate_daily(news)
    else:
        scored = score_news(news)
        scored.to_csv(PROC_DIR / "sentiment_data.csv", index=False)
        daily = aggregate_daily(scored)

    final = build_final_dataset(feats, daily)  # restricted to the news window when news exists
    final.to_csv(PROC_DIR / "final_dataset.csv", index=False)
    cov = sentiment_coverage(final)
    print(f"final_dataset.csv: {cov['rows']} rows, {cov['days_with_news']} with news ({cov['coverage']:.0%})")
    return final


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("ticker", nargs="?", default=DEFAULT_TICKER); b.add_argument("--news-csv")
    t = sub.add_parser("train"); t.add_argument("--allow-price-only", action="store_true", help="train even if news is too sparse for the sentiment model")
    pr = sub.add_parser("predict"); pr.add_argument("ticker", nargs="?", default=DEFAULT_TICKER)
    llm = sub.add_parser("llm-predict")
    llm.add_argument("ticker", nargs="?", default=DEFAULT_TICKER)
    llm.add_argument("--dummy", action="store_true", default=True, help="Use realistic synthetic dummy data")
    llm.add_argument("--trend", default="bullish", choices=["bullish", "bearish", "neutral", "volatile"], help="Dummy data trend")
    llm.add_argument("--eval", action="store_true", help="Run backtest evaluation over dummy test set")
    sub.add_parser("serve")
    a = p.parse_args()

    if a.cmd == "build":
        build(a.ticker, a.news_csv)
    elif a.cmd == "train":
        from src.models.train import train
        train(allow_price_only=a.allow_price_only)
    elif a.cmd == "predict":
        import json
        from src.models.predict import predict_ticker
        print(json.dumps(predict_ticker(a.ticker.upper()), indent=2))
    elif a.cmd == "llm-predict":
        from src.models.llm_predictor import LLMStockPredictor
        predictor = LLMStockPredictor(provider="auto")
        if a.eval:
            result = predictor.evaluate_on_dummy_data(days=45, window_size=15, ticker=a.ticker.upper(), trend=a.trend)
            print(f"\n--- Backtest Evaluation on Dummy Data ({a.ticker.upper()}, {a.trend}) ---")
            print(f"Total Test Days:      {result['total_test_days']}")
            print(f"Directional Accuracy: {result['directional_accuracy_pct']}%")
            print(f"MAE:                  ${result['mae']:.2f}")
            print(f"RMSE:                 ${result['rmse']:.2f}")
            print(f"MAPE:                 {result['mape_pct']:.2f}%")
        else:
            prices_df, news_df = predictor.generate_dummy_data(days=60, ticker=a.ticker.upper(), trend=a.trend)
            pred = predictor.predict(prices_df, news_df, ticker=a.ticker.upper())
            print(f"\n--- LLM Stock Price Prediction ({pred.provider}) ---")
            print(f"Ticker:                 {pred.ticker}")
            print(f"Current Price:          ${pred.current_price:.2f}")
            print(f"Predicted Target Price: ${pred.predicted_target_price:.2f}")
            print(f"Predicted Direction:    {pred.predicted_direction}")
            print(f"Expected Change:        {pred.predicted_change_percent:+.2f}%")
            print(f"Confidence:             {pred.confidence * 100:.1f}%")
            print(f"\n[Reasoning]:\n{pred.reasoning}")
            print(f"\n[Technical Summary]:\n{pred.technical_summary}")
            print(f"\n[Sentiment Summary]:\n{pred.sentiment_summary}")
            print(f"\n[Risk Factors]:")
            for rf in pred.risk_factors:
                print(f"  - {rf}")
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
