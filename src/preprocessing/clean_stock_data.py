import pandas as pd


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    return df.reset_index(drop=True)
