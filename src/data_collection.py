import yfinance as yf
import pandas as pd

ticker = "RELIANCE.NS"

data = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=False
)

# Remove empty rows
data.dropna(inplace=True)

# Save dataset
data.to_csv("RELIANCE.csv")

print("Dataset downloaded successfully!")
print(data.head())
print("\nShape:", data.shape)