# sentiment_finbert.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datetime import datetime

# GPU / CPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "yiyanghkust/finbert-tone"  # Pretrained FinBERT

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def get_sentiment(text):
    """
    Returns FinBERT sentiment score: (positive - negative)
    """
    if not isinstance(text, str) or text.strip() == "":
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    # labels → 0=neutral, 1=positive, 2=negative
    score = float(probs[1] - probs[2])
    return score

def run_finbert(news_csv="TSLA_news.csv"):
    print("\n🚀 Running FinBERT sentiment analysis...\n")

    df = pd.read_csv(news_csv)

    # Combine text fields
    df["text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["content"].fillna("")
    )

    sentiment_scores = []

    for text in tqdm(df["text"], desc="Processing articles"):
        sentiment_scores.append(get_sentiment(text))

    df["sentiment"] = sentiment_scores

    # Extract date only
    df["date"] = pd.to_datetime(df["publishedAt"]).dt.date

    # Aggregate by date → daily sentiment
    daily = df.groupby("date")["sentiment"].mean().reset_index()

    daily.to_csv("TSLA_sentiment.csv", index=False)
    print("\n✅ Saved: TSLA_sentiment.csv")

    return daily


if __name__ == "__main__":
    run_finbert()
