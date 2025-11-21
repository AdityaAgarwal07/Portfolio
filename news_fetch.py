from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "11b4b6696b8b4d7a8db8fb655089eeb3"
newsapi = NewsApiClient(api_key=API_KEY)

def fetch_news(ticker="TSLA", days=30):
    query = f'"{ticker}" OR "{ticker} stock" OR "{ticker} shares"'
    
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days)

    print(f"\nFetching news for {ticker}...\n")

    articles = newsapi.get_everything(
        q=query,
        language='en',
        sort_by='publishedAt',
        from_param=from_date.strftime('%Y-%m-%d'),
        to=to_date.strftime('%Y-%m-%d'),
        page_size=100
    )

    data = []
    for a in articles.get("articles", []):
        data.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url")
        })
    
    df = pd.DataFrame(data)
    df.to_csv("TSLA_news.csv", index=False)

    print("Saved TSLA_news.csv")
    return df


if __name__ == "__main__":
    fetch_news()
