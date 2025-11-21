import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def get_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    score = float(probs[1] - probs[2])  # (positive - negative)
    return score

def run_finbert(news_file="TSLA_news.csv"):
    print("\nRunning FinBERT...\n")

    df = pd.read_csv(news_file)

    df["full_text"] = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["content"].fillna("")
    )

    scores = []
    for text in tqdm(df["full_text"], desc="Scoring"):
        scores.append(get_sentiment(text))

    df["sentiment_score"] = scores
    df["date"] = pd.to_datetime(df["publishedAt"]).dt.date

    df.to_csv("TSLA_news_with_sentiment.csv", index=False)

    print("\nSaved: TSLA_news_with_sentiment.csv")
    return df

if __name__ == "__main__":
    run_finbert()
