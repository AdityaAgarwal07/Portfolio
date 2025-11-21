# prepare_dataset.py

import pandas as pd
import numpy as np

# ---------------------- TECHNICAL INDICATORS ----------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def add_technical_features(df):
    df["return_1d"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()

    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["vol_10"] = df["return_1d"].rolling(10).std()
    df["vol_20"] = df["return_1d"].rolling(20).std()

    df["rsi_14"] = compute_rsi(df["Close"])

    return df


# ---------------------- MAIN MERGE FUNCTION ----------------------

def prepare_dataset(price_csv="TSLA_price.csv", sentiment_csv="TSLA_sentiment.csv"):
    print("\n📌 Preparing final merged dataset...\n")

    # ---------------------- FIX PRICE FILE ----------------------
    price = pd.read_csv(price_csv)

    # Remove bad rows like: ["Ticker", "TSLA", "TSLA", ...]
    price = price[~price["Price"].astype(str).str.contains("Ticker")]

    # Rename first column
    price.rename(columns={"Price": "Date"}, inplace=True)

    # Convert Date column
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        price[col] = pd.to_numeric(price[col], errors="coerce")

    # Drop remaining bad rows
    price.dropna(inplace=True)

    # Create a date column for merging
    price["date"] = price["Date"].dt.date


    # ---------------------- LOAD SENTIMENT ----------------------
    sentiment = pd.read_csv(sentiment_csv)
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date


    # ---------------------- MERGE ----------------------
    df = price.merge(sentiment, on="date", how="left")

    # Fill sentiment missing as 0
    df["sentiment"] = df["sentiment"].fillna(0)

    # ---------------------- ADD TECHNICAL FEATURES ----------------------
    df = add_technical_features(df)

    # ---------------------- CREATE TARGET ----------------------
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)

    # Final cleaning
    df.dropna(inplace=True)

    # Save final dataset
    df.to_csv("TSLA_dataset.csv", index=False)
    print("✅ Saved: TSLA_dataset.csv")

    return df


if __name__ == "__main__":
    prepare_dataset()
