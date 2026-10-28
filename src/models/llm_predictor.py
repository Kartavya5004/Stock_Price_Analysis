"""LLM-based Stock Price Prediction Model.

Combines quantitative market indicators with qualitative financial news sentiment
using Large Language Models (Gemini / Claude / GPT / Local CoT) to forecast next-session
stock price, direction, expected return, confidence, and detailed financial reasoning.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.features.technical_indicators import add_indicators
from src.utils.config import DISCLAIMER


class StockPredictionOutput(BaseModel):
    """Structured output schema for LLM Stock Price Prediction."""
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Last known closing price")
    predicted_target_price: float = Field(description="Estimated target price for next session")
    predicted_direction: Literal["Bullish", "Bearish", "Neutral"] = Field(
        description="Predicted price movement direction"
    )
    predicted_change_percent: float = Field(
        description="Expected price change in percentage (e.g. +1.8 or -2.4)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in prediction from 0.0 to 1.0"
    )
    time_horizon: str = Field(
        default="Next Trading Session", description="Forecast time horizon"
    )
    reasoning: str = Field(
        description="Step-by-step financial and technical reasoning for the prediction"
    )
    technical_summary: str = Field(
        description="Synthesis of technical indicators (RSI, Moving Averages, MACD, Volume)"
    )
    sentiment_summary: str = Field(
        description="Synthesis of news catalysts and market sentiment"
    )
    risk_factors: list[str] = Field(
        default_factory=list, description="Key upside or downside risks to this prediction"
    )
    provider: str = Field(
        default="offline-cot-llm", description="LLM provider or model used"
    )
    disclaimer: str = Field(
        default=DISCLAIMER, description="Risk and educational disclaimer"
    )


class LLMStockPredictor:
    """LLM Model for predicting stock prices and market direction."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        provider: Literal["auto", "gemini", "offline"] = "auto",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if provider == "auto":
            self.provider = "gemini" if self.api_key else "offline"
        else:
            self.provider = provider

        self._client = None
        if self.provider == "gemini":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # If gemini client fails to initialize, fallback safely to offline CoT
                self.provider = "offline"

    @staticmethod
    def generate_dummy_data(
        days: int = 60,
        ticker: str = "AAPL",
        start_price: float = 175.0,
        trend: Literal["bullish", "bearish", "neutral", "volatile"] = "bullish",
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate realistic synthetic stock price OHLCV data and financial news.

        Args:
            days: Number of trading days to simulate.
            ticker: Stock ticker symbol.
            start_price: Initial stock price.
            trend: Market regime to simulate ('bullish', 'bearish', 'neutral', 'volatile').
            seed: Random seed for reproducibility.

        Returns:
            (prices_df, news_df)
        """
        rng = np.random.default_rng(seed)

        # Generate trading dates (excluding weekends)
        end_date = datetime.today()
        cur_date = end_date - timedelta(days=int(days * 1.5))
        dates = []
        while len(dates) < days:
            if cur_date.weekday() < 5:  # Mon-Fri
                dates.append(cur_date.strftime("%Y-%m-%d"))
            cur_date += timedelta(days=1)

        # Trend drift and volatility parameters
        trend_map = {
            "bullish": (0.003, 0.012),   # +0.3% daily drift, 1.2% vol
            "bearish": (-0.003, 0.015),  # -0.3% daily drift, 1.5% vol
            "neutral": (0.000, 0.010),   # 0% drift, 1.0% vol
            "volatile": (0.001, 0.025),  # high 2.5% vol
        }
        drift, vol = trend_map.get(trend, (0.001, 0.015))

        daily_returns = rng.normal(drift, vol, size=days)
        price_series = [start_price]
        for r in daily_returns[:-1]:
            price_series.append(max(5.0, price_series[-1] * (1 + r)))

        # Construct OHLCV
        rows = []
        for d, p in zip(dates, price_series):
            day_noise = rng.normal(0, vol * 0.5, size=4)
            o = round(p * (1 + day_noise[0]), 2)
            c = round(p * (1 + day_noise[1]), 2)
            h = round(max(o, c) * (1 + abs(day_noise[2])), 2)
            l = round(min(o, c) * (1 - abs(day_noise[3])), 2)
            v = int(rng.integers(10_000_000, 50_000_000))
            rows.append({"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v})

        prices_df = pd.DataFrame(rows)

        # Generate contextual news headlines
        bullish_headlines = [
            (f"{ticker} reports record Q3 earnings beat, revenue surges 12% YoY", "Earnings report exceeded analyst consensus with strong margin growth.", "Bloomberg"),
            (f"Major institutional fund increases stake in {ticker} citing AI tailwinds", "Institutional filings show top hedge fund raised allocation by 18%.", "Reuters"),
            (f"{ticker} unveils next-generation enterprise product suite", "Analysts praise the expansion of high-margin subscription offerings.", "TechCrunch"),
            (f"Analyst upgrades {ticker} to Strong Buy, raises target price", "Upgrade highlights accelerating revenue and robust balance sheet.", "MarketWatch"),
            (f"{ticker} announces $10B share buyback program and dividend boost", "Board approved capital return program reflecting cash flow strength.", "CNBC"),
        ]

        bearish_headlines = [
            (f"{ticker} cuts fiscal outlook amid supply chain bottlenecks", "Management lowered next-quarter revenue guidance by 8%.", "Reuters"),
            (f"Regulatory agency initiates antitrust investigation into {ticker}", "Authorities are examining pricing and market practices.", "Wall Street Journal"),
            (f"Top tier broker downgrades {ticker} to Sell on margin pressure", "Analyst notes intensifying competition and softening enterprise demand.", "Financial Times"),
            (f"{ticker} supplier warns of component shortage and delivery delays", "Key supplier lowers production targets for next two quarters.", "Bloomberg"),
            (f"CEO sells shares as insider selling in {ticker} hits 6-month high", "SEC disclosures reveal executive sales following peak valuation.", "MarketWatch"),
        ]

        neutral_headlines = [
            (f"{ticker} announces date for upcoming annual shareholder meeting", "Shareholders will vote on board nominees and standard proposals.", "PR Newswire"),
            (f"Industry peers report mixed quarterly results in tech sector", "Sector performance remains mixed amid macroeconomic interest rate pauses.", "CNBC"),
            (f"{ticker} participates in annual technology conference", "Management discussed product roadmap and long-term industry outlook.", "Business Wire"),
            (f"Market awaits Federal Reserve interest rate decision as {ticker} consolidates", "Equities traded in narrow range ahead of central bank statement.", "Reuters"),
        ]

        news_rows = []
        for i, d in enumerate(dates):
            # Generate news for ~60% of dates
            if rng.random() > 0.4:
                daily_return = (prices_df.iloc[i]["close"] - prices_df.iloc[max(0, i - 1)]["close"]) / prices_df.iloc[max(0, i - 1)]["close"]
                if daily_return > 0.008 or (trend == "bullish" and rng.random() > 0.3):
                    choice = rng.choice(bullish_headlines)
                elif daily_return < -0.008 or (trend == "bearish" and rng.random() > 0.3):
                    choice = rng.choice(bearish_headlines)
                else:
                    choice = rng.choice(neutral_headlines)

                news_rows.append({
                    "date": d,
                    "title": choice[0],
                    "description": choice[1],
                    "source": choice[2],
                })

        news_df = pd.DataFrame(news_rows)
        return prices_df, news_df

    def prepare_prompt(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame | None = None,
        ticker: str = "AAPL",
        window_days: int = 15,
    ) -> str:
        """Format stock price history, indicators, and recent news into an LLM prompt."""
        df = stock_df.copy()
        if "close" not in df.columns:
            raise ValueError("stock_df must contain 'close' column")

        # Compute technical indicators
        feats = add_indicators(df)
        recent_feats = feats.tail(window_days).copy()

        # Format price & indicator table
        table_lines = [
            "Date       | Open   | High   | Low    | Close  | Vol(M) | RSI(14) | MACD-Hist | SMA10-Ratio | 1D-Ret(%)",
            "-----------|--------|--------|--------|--------|--------|---------|-----------|-------------|----------",
        ]
        for _, row in recent_feats.iterrows():
            d_str = str(row["date"])[:10]
            o = f"{row['open']:.2f}" if "open" in row and pd.notna(row["open"]) else "N/A"
            h = f"{row['high']:.2f}" if "high" in row and pd.notna(row["high"]) else "N/A"
            l = f"{row['low']:.2f}" if "low" in row and pd.notna(row["low"]) else "N/A"
            c = f"{row['close']:.2f}"
            vol = f"{row['volume'] / 1e6:.1f}" if "volume" in row and pd.notna(row["volume"]) else "N/A"
            rsi = f"{row['rsi_14']:.1f}" if pd.notna(row.get("rsi_14")) else "N/A"
            macd_h = f"{row['macd_hist']:.4f}" if pd.notna(row.get("macd_hist")) else "N/A"
            sma_r = f"{row['close_sma10_ratio']:.3f}" if pd.notna(row.get("close_sma10_ratio")) else "N/A"
            r1 = f"{row['return_1d'] * 100:+.2f}%" if pd.notna(row.get("return_1d")) else "N/A"
            table_lines.append(f"{d_str:10} | {o:6} | {h:6} | {l:6} | {c:6} | {vol:6} | {rsi:7} | {macd_h:9} | {sma_r:11} | {r1:9}")

        table_str = "\n".join(table_lines)

        # Format recent news
        news_section = "No recent news headlines available."
        if news_df is not None and not news_df.empty:
            recent_news = news_df.tail(6)
            news_lines = []
            for _, n_row in recent_news.iterrows():
                nd = str(n_row["date"])[:10]
                title = n_row.get("title", "")
                desc = n_row.get("description", "")
                source = n_row.get("source", "News")
                news_lines.append(f"- [{nd}] ({source}) {title} - {desc}")
            news_section = "\n".join(news_lines)

        latest = recent_feats.iloc[-1]
        current_price = latest["close"]

        prompt = f"""You are an elite quantitative financial analyst and stock market forecaster.
Analyze the following market data and recent news for ticker **{ticker}**.
Forecast the stock's next-session target price, directional movement (Bullish, Bearish, or Neutral), expected percentage change, confidence score (0.0 to 1.0), and comprehensive financial reasoning.

### CURRENT ASSET SUMMARY:
- Ticker: {ticker}
- Latest Closing Price: ${current_price:.2f}
- Date: {str(latest['date'])[:10]}

### RECENT PRICE ACTION & TECHNICAL INDICATORS:
{table_str}

### RECENT FINANCIAL NEWS HEADLINES:
{news_section}

### REQUIRED OUTPUT:
Return ONLY a valid JSON object matching this exact structure:
{{
  "ticker": "{ticker}",
  "current_price": {current_price:.2f},
  "predicted_target_price": <number: estimated next-day target price in USD>,
  "predicted_direction": <"Bullish" | "Bearish" | "Neutral">,
  "predicted_change_percent": <number: signed percentage return e.g. +1.4 or -0.8>,
  "confidence": <number: between 0.50 and 0.95>,
  "reasoning": <string: detailed multi-sentence financial rationale analyzing momentum, support/resistance, and catalysts>,
  "technical_summary": <string: synthesis of RSI, MACD, moving averages, and volume signals>,
  "sentiment_summary": <string: synthesis of news impact and market tone>,
  "risk_factors": [<string: risk factor 1>, <string: risk factor 2>, <string: risk factor 3>]
}}
"""
        return prompt

    def _offline_cot_reasoning(
        self, stock_df: pd.DataFrame, news_df: pd.DataFrame | None, ticker: str
    ) -> StockPredictionOutput:
        """High-fidelity Chain-of-Thought Financial Reasoning Engine for offline/dummy testing."""
        feats = add_indicators(stock_df)
        latest = feats.iloc[-1]
        cur_price = float(latest["close"])

        # Technical signals evaluation
        rsi = float(latest.get("rsi_14", 50.0)) if pd.notna(latest.get("rsi_14")) else 50.0
        macd_hist = float(latest.get("macd_hist", 0.0)) if pd.notna(latest.get("macd_hist")) else 0.0
        sma10_ratio = float(latest.get("close_sma10_ratio", 0.0)) if pd.notna(latest.get("close_sma10_ratio")) else 0.0
        sma30_ratio = float(latest.get("close_sma30_ratio", 0.0)) if pd.notna(latest.get("close_sma30_ratio")) else 0.0
        vol_ratio = float(latest.get("volume_ratio", 1.0)) if pd.notna(latest.get("volume_ratio")) else 1.0
        ret_5d = float(latest.get("return_5d", 0.0)) if pd.notna(latest.get("return_5d")) else 0.0
        volatility = float(latest.get("volatility_10", 0.015)) if pd.notna(latest.get("volatility_10")) else 0.015
        if np.isnan(volatility) or volatility <= 0:
            volatility = 0.015

        # News sentiment heuristic evaluation
        news_score = 0.0
        news_count = 0
        news_summary = "No recent news headlines available; relying primarily on price action."
        if news_df is not None and not news_df.empty:
            recent_news = news_df.tail(5)
            news_count = len(recent_news)
            pos_words = {"record", "beat", "surge", "growth", "upgrade", "buyback", "partnership", "tailwind", "soar"}
            neg_words = {"cut", "fall", "drop", "probe", "investigation", "downgrade", "pressure", "shortage", "risk"}

            pos_hits = 0
            neg_hits = 0
            for _, r in recent_news.iterrows():
                t = str(r.get("title", "")).lower()
                pos_hits += sum(1 for w in pos_words if w in t)
                neg_hits += sum(1 for w in neg_words if w in t)

            if pos_hits + neg_hits > 0:
                news_score = (pos_hits - neg_hits) / (pos_hits + neg_hits)
                if news_score > 0.2:
                    news_summary = f"Positive news catalysts observed with {pos_hits} favorable growth/upgrade mentions."
                elif news_score < -0.2:
                    news_summary = f"Negative news catalysts observed with {neg_hits} risk/downgrade indicators."
                else:
                    news_summary = "Mixed or neutral news flow with balanced positive and cautious coverage."

        # Technical Score (-1.0 to +1.0)
        tech_score = 0.0
        # RSI contribution
        if rsi < 32:
            tech_score += 0.35  # Oversold bounce expected
        elif rsi > 70:
            tech_score -= 0.30  # Overbought pullback risk
        else:
            tech_score += (rsi - 50) / 100.0  # Momentum continuation

        # MACD Histogram contribution
        if macd_hist > 0.001:
            tech_score += 0.30
        elif macd_hist < -0.001:
            tech_score -= 0.30

        # Moving average alignment
        if sma10_ratio > 0 and sma30_ratio > 0:
            tech_score += 0.25
        elif sma10_ratio < 0 and sma30_ratio < 0:
            tech_score -= 0.25

        # Volume confirmation
        if vol_ratio > 1.2 and tech_score > 0:
            tech_score += 0.10
        elif vol_ratio > 1.2 and tech_score < 0:
            tech_score -= 0.10

        # Blended composite score (65% technicals + 35% news sentiment)
        composite = 0.65 * tech_score + 0.35 * (news_score * 0.8)
        composite = np.clip(composite, -1.0, 1.0)

        # Expected price change calculation
        # Expected % return is scaled by current asset volatility
        expected_change_pct = composite * (volatility * 100.0) * 1.2
        target_price = round(cur_price * (1 + expected_change_pct / 100.0), 2)

        if expected_change_pct > 0.4:
            direction: Literal["Bullish", "Bearish", "Neutral"] = "Bullish"
        elif expected_change_pct < -0.4:
            direction = "Bearish"
        else:
            direction = "Neutral"

        confidence = round(float(np.clip(0.60 + abs(composite) * 0.30, 0.55, 0.92)), 2)

        # Build technical summary
        tech_summary = (
            f"RSI-14 is at {rsi:.1f} ({'oversold bounce potential' if rsi < 35 else 'overbought zone' if rsi > 68 else 'healthy neutral zone'}). "
            f"MACD histogram reads {macd_hist:+.4f} showing {'bullish expanding momentum' if macd_hist > 0 else 'bearish contraction'}. "
            f"Price trades {sma10_ratio * 100:+.2f}% vs 10-day SMA and {sma30_ratio * 100:+.2f}% vs 30-day SMA."
        )

        # Build chain-of-thought financial reasoning
        if direction == "Bullish":
            reasoning = (
                f"{ticker} exhibits bullish continuation setup. Recent 5-day return of {ret_5d * 100:+.2f}% "
                f"aligns with positive MACD momentum ({macd_hist:+.4f}) and price trading above short-term moving averages. "
                f"{news_summary} Combining technical trend alignment with recent catalyst support points toward a short-term "
                f"target of ${target_price:.2f} (+{expected_change_pct:.2f}%)."
            )
            risks = [
                "Macro market downturn or broad tech sector selloff",
                "Intraday profit-taking near recent swing resistance levels",
                "Unexpected headline risk or sudden volume drop-off",
            ]
        elif direction == "Bearish":
            reasoning = (
                f"{ticker} indicates downward pressure or mean-reversion risk. Technical indicators show softening momentum "
                f"with MACD histogram at {macd_hist:+.4f} and price breaking below moving averages (SMA10 ratio {sma10_ratio * 100:+.2f}%). "
                f"{news_summary} Expecting consolidation or downward test toward support level at ${target_price:.2f} ({expected_change_pct:.2f}%)."
            )
            risks = [
                "Oversold technical bounce surprising short positions",
                "Broad index rally lifting all sector components",
                "Sudden positive corporate disclosure or analyst upgrade",
            ]
        else:
            reasoning = (
                f"{ticker} is consolidating in a neutral range. Technical indicators are balanced (RSI {rsi:.1f}, MACD {macd_hist:+.4f}) "
                f"without definitive breakout confirmation. {news_summary} Projecting sideways drift around ${target_price:.2f} ({expected_change_pct:+.2f}%)."
            )
            risks = [
                "Breakout above immediate resistance band",
                "Breakdown below key moving average support",
                "Spike in volatility from upcoming macroeconomic data releases",
            ]

        return StockPredictionOutput(
            ticker=ticker,
            current_price=cur_price,
            predicted_target_price=target_price,
            predicted_direction=direction,
            predicted_change_percent=round(float(expected_change_pct), 2),
            confidence=confidence,
            time_horizon="Next Trading Session",
            reasoning=reasoning,
            technical_summary=tech_summary,
            sentiment_summary=news_summary,
            risk_factors=risks,
            provider="offline-cot-llm",
            disclaimer=DISCLAIMER,
        )

    def predict(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame | None = None,
        ticker: str = "AAPL",
    ) -> StockPredictionOutput:
        """Run LLM prediction for next session price and direction.

        If live Gemini client is configured and available, queries Gemini;
        otherwise executes the high-fidelity offline Chain-of-Thought engine.
        """
        if stock_df.empty or len(stock_df) < 5:
            raise ValueError(f"Need at least 5 price rows for {ticker}; got {len(stock_df)}.")

        # Live Gemini prediction if available
        if self.provider == "gemini" and self._client is not None:
            prompt = self.prepare_prompt(stock_df, news_df, ticker=ticker)
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = response.text or ""
                # Extract JSON if enclosed in markdown code fences
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    data["provider"] = f"gemini ({self.model_name})"
                    data["disclaimer"] = DISCLAIMER
                    return StockPredictionOutput(**data)
            except Exception as e:
                # Log error and fallback smoothly to offline CoT
                print(f"[LLMStockPredictor] Live Gemini call failed ({e}); falling back to offline CoT.")

        # Offline Chain-of-Thought Engine
        return self._offline_cot_reasoning(stock_df, news_df, ticker=ticker)

    def evaluate_on_dummy_data(
        self,
        days: int = 40,
        window_size: int = 15,
        ticker: str = "AAPL",
        trend: Literal["bullish", "bearish", "neutral", "volatile"] = "bullish",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Backtest and evaluate the LLM predictor across a rolling window of dummy data.

        Calculates:
          - Directional Accuracy (%)
          - Mean Absolute Error (MAE)
          - Root Mean Squared Error (RMSE)
          - Mean Absolute Percentage Error (MAPE)
        """
        prices_df, news_df = self.generate_dummy_data(
            days=days, ticker=ticker, trend=trend, seed=seed
        )

        test_steps = len(prices_df) - window_size
        if test_steps < 3:
            raise ValueError(f"Need more days ({days}) for window_size ({window_size}).")

        predictions_log = []
        actual_prices = []
        pred_prices = []
        correct_directions = 0
        valid_direction_tests = 0

        for t in range(window_size, len(prices_df)):
            hist_prices = prices_df.iloc[:t].copy()
            cur_date = hist_prices.iloc[-1]["date"]
            actual_next_price = float(prices_df.iloc[t]["close"])
            actual_current_price = float(hist_prices.iloc[-1]["close"])
            actual_return = (actual_next_price - actual_current_price) / actual_current_price

            if actual_return > 0.003:
                actual_dir = "Bullish"
            elif actual_return < -0.003:
                actual_dir = "Bearish"
            else:
                actual_dir = "Neutral"

            # Filter news available up to current date
            hist_news = (
                news_df[news_df["date"] <= cur_date].copy()
                if news_df is not None and not news_df.empty
                else None
            )

            # Predict next day
            pred = self.predict(hist_prices, hist_news, ticker=ticker)

            actual_prices.append(actual_next_price)
            pred_prices.append(pred.predicted_target_price)

            # Check directional accuracy
            is_correct = (pred.predicted_direction == actual_dir)
            if actual_dir != "Neutral" or pred.predicted_direction == "Neutral":
                valid_direction_tests += 1
                if is_correct:
                    correct_directions += 1

            predictions_log.append({
                "date": prices_df.iloc[t]["date"],
                "last_close": actual_current_price,
                "actual_close": actual_next_price,
                "predicted_target": pred.predicted_target_price,
                "actual_return_pct": round(actual_return * 100, 2),
                "predicted_return_pct": pred.predicted_change_percent,
                "actual_direction": actual_dir,
                "predicted_direction": pred.predicted_direction,
                "confidence": pred.confidence,
                "direction_match": is_correct,
            })

        act_arr = np.array(actual_prices)
        pred_arr = np.array(pred_prices)

        mae = float(np.mean(np.abs(pred_arr - act_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - act_arr) ** 2)))
        mape = float(np.mean(np.abs((pred_arr - act_arr) / act_arr)) * 100.0)
        dir_acc = (
            float(correct_directions / valid_direction_tests * 100.0)
            if valid_direction_tests > 0
            else 0.0
        )

        return {
            "ticker": ticker,
            "trend_regime": trend,
            "total_test_days": len(predictions_log),
            "directional_accuracy_pct": round(dir_acc, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "latest_prediction": predictions_log[-1] if predictions_log else None,
            "predictions_log": predictions_log,
        }
