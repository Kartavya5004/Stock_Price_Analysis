"""Test and demonstrate the LLM Stock Price Prediction Model using dummy data.

Usage:
  python test_llm_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.llm_predictor import LLMStockPredictor


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def test_single_prediction(predictor: LLMStockPredictor, ticker: str, trend: str):
    print_banner(f"1. Single Prediction Test: {ticker} ({trend.upper()} Market Regime)")

    print(f"[*] Generating synthetic 60-day stock history and news for {ticker}...")
    prices_df, news_df = predictor.generate_dummy_data(
        days=60, ticker=ticker, start_price=175.0, trend=trend, seed=42
    )

    print(f"[*] Generated {len(prices_df)} price bars and {len(news_df)} news headlines.")
    print("\n--- Recent 5 Days Price Data ---")
    print(prices_df.tail(5)[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))

    print("\n--- Recent 3 News Headlines ---")
    for _, r in news_df.tail(3).iterrows():
        print(f"  • [{r['date']}] ({r['source']}) {r['title']}")

    print(f"\n[*] Running LLM Stock Price Prediction ({predictor.provider})...")
    prediction = predictor.predict(prices_df, news_df, ticker=ticker)

    print_banner(f"LLM Prediction Output for {ticker}")
    print(f"  Ticker:                 {prediction.ticker}")
    print(f"  Current Price:          ${prediction.current_price:.2f}")
    print(f"  Predicted Target Price: ${prediction.predicted_target_price:.2f}")
    print(f"  Predicted Direction:    {prediction.predicted_direction.upper()}")
    print(f"  Expected Change:        {prediction.predicted_change_percent:+.2f}%")
    print(f"  Confidence:             {prediction.confidence * 100:.1f}%")
    print(f"  Provider Engine:        {prediction.provider}")
    print(f"\n  [Technical Analysis]:\n  {prediction.technical_summary}")
    print(f"\n  [News Sentiment Analysis]:\n  {prediction.sentiment_summary}")
    print(f"\n  [Financial Reasoning (Chain of Thought)]:\n  {prediction.reasoning}")
    print(f"\n  [Key Risk Factors]:")
    for risk in prediction.risk_factors:
        print(f"    - {risk}")
    print(f"\n  [Disclaimer]:\n  {prediction.disclaimer}")


def test_backtest_evaluation(predictor: LLMStockPredictor, ticker: str, trend: str):
    print_banner(f"2. Multi-Day Backtest Evaluation on Dummy Data ({ticker})")

    print(f"[*] Running rolling-window backtest over 30 test days...")
    eval_result = predictor.evaluate_on_dummy_data(
        days=45, window_size=15, ticker=ticker, trend=trend, seed=101
    )

    log = eval_result["predictions_log"]
    print(f"[*] Evaluated {len(log)} consecutive trading sessions.\n")

    print(f"{'Date':10} | {'Last Close':10} | {'Predicted':10} | {'Actual Close':12} | {'Pred Dir':10} | {'Act Dir':10} | {'Match?'}")
    print("-" * 80)
    for row in log[-10:]:  # Display last 10 test days
        match_str = "YES" if row["direction_match"] else "NO"
        print(
            f"{row['date']:10} | "
            f"${row['last_close']:<9.2f} | "
            f"${row['predicted_target']:<9.2f} | "
            f"${row['actual_close']:<11.2f} | "
            f"{row['predicted_direction']:<10} | "
            f"{row['actual_direction']:<10} | "
            f"{match_str}"
        )

    print("\n--- Summary Performance Metrics ---")
    print(f"  Directional Accuracy: {eval_result['directional_accuracy_pct']}%")
    print(f"  Mean Absolute Error (MAE): ${eval_result['mae']:.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${eval_result['rmse']:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {eval_result['mape_pct']:.2f}%")


def main():
    print("=" * 70)
    print("    STOCK PRICE ANALYSIS - LLM PREDICTION ENGINE TEST RUNNER")
    print("=" * 70)

    # Initialize predictor (will automatically use Gemini if API key is present, or offline CoT engine)
    predictor = LLMStockPredictor(provider="auto")
    print(f"Initialized LLM Stock Predictor in mode: [{predictor.provider.upper()}]")

    # 1. Bullish scenario test
    test_single_prediction(predictor, ticker="AAPL", trend="bullish")

    # 2. Bearish scenario test
    test_single_prediction(predictor, ticker="TSLA", trend="bearish")

    # 3. Rolling Backtest Evaluation
    test_backtest_evaluation(predictor, ticker="AAPL", trend="bullish")

    print("\n" + "=" * 70)
    print(" ALL DUMMY DATA TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
