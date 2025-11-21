# price_fetch.py

import yfinance as yf
import pandas as pd

def fetch_price(ticker="TSLA", start="2020-01-01"):
    print(f"\nFetching price data for {ticker}...\n")
    
    df = yf.download(ticker, start=start)
    
    # Keep only important columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df.to_csv(f"{ticker}_price.csv")
    
    print(f"Saved {ticker}_price.csv")
    return df

if __name__ == "__main__":
    fetch_price("TSLA")
