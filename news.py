# news_fetcher.py - Real NewsAPI.org Integration for Indian Stocks

"""
Fetch real news from NewsAPI.org for ALL Indian stocks
- Fetches 200 articles from last 2 days
- Analyzes sentiment
- Maps news to stocks
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
from config import NEWSAPI_KEY_1, NEWSAPI_KEY_2, NEWS_LOOKBACK_DAYS, NEWS_MAX_ARTICLES, POPULAR_INDIAN_STOCKS, INDIAN_STOCKS

class NewsAPIFetcher:
    """Fetch real news from NewsAPI.org"""
    
    def __init__(self):
        self.api_keys = []
        if NEWSAPI_KEY_1:
            self.api_keys.append(NEWSAPI_KEY_1)
        if NEWSAPI_KEY_2:
            self.api_keys.append(NEWSAPI_KEY_2)
        
        self.current_key_index = 0
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Company name mapping (symbol to company name)
        self.company_names = {
            'SBIN.NS': 'State Bank of India', 'PNB.NS': 'Punjab National Bank',
            'CANBK.NS': 'Canara Bank', 'INOXWIND.NS': 'Inox Wind',
            'SUZLON.NS': 'Suzlon Energy', 'BSE.NS': 'BSE Limited',
            'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys', 'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank', 'TATAMOTORS.NS': 'Tata Motors',
            'GSPL.NS': 'Gujarat State Petronet', 'EPL.NS': 'EPL Limited',
            'UNIPARTS.NS': 'Uniparts India', 'UNIMECH.NS': 'Unimech Aerospace',
            'ZOMATO.NS': 'Zomato', 'PAYTM.NS': 'Paytm', 'NYKAA.NS': 'Nykaa',
            # Add more as needed
        }
        
        print(f"✅ NewsAPI Fetcher initialized with {len(self.api_keys)} API key(s)")
    
    def get_api_key(self):
        """Get current API key"""
        if not self.api_keys:
            return None
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    def fetch_indian_stock_news(self, max_articles=200):
        """
        Fetch news for Indian stocks from NewsAPI.org
        Returns: DataFrame with news and sentiment
        """
        print(f"📰 Fetching {max_articles} news articles from NewsAPI.org...")
        
        if not self.api_keys:
            print("⚠️ No API keys configured. Using demo mode.")
            return self.generate_demo_news(max_articles)
        
        all_news = []
        
        # Calculate date range
        to_date = datetime.now(self.ist)
        from_date = to_date - timedelta(days=NEWS_LOOKBACK_DAYS)
        
        # Search queries for Indian stocks
        search_queries = [
            'Indian stock market',
            'NSE BSE',
            'Indian shares',
            'Mumbai stock exchange',
            'Nifty Sensex',
            'Indian banking stocks',
            'Indian IT stocks',
            'Indian pharma stocks',
            'Indian renewable energy',
            'Indian PSU banks'
        ]
        
        articles_per_query = max_articles // len(search_queries)
        
        for query in search_queries:
            try:
                api_key = self.get_api_key()
                
                url = 'https://newsapi.org/v2/everything'
                params = {
                    'q': query,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': articles_per_query,
                    'apiKey': api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        all_news.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'query': query
                        })
                    
                    print(f"✅ Fetched {len(articles)} articles for '{query}'")
                
                elif response.status_code == 429:
                    print(f"⚠️ Rate limit reached for '{query}'")
                    break
                
                else:
                    print(f"⚠️ Error {response.status_code} for '{query}'")
            
            except Exception as e:
                print(f"⚠️ Error fetching '{query}': {e}")
                continue
        
        if not all_news:
            print("⚠️ No news fetched. Using demo mode.")
            return self.generate_demo_news(max_articles)
        
        # Process news
        df = pd.DataFrame(all_news)
        df = self.map_news_to_stocks(df)
        df = self.analyze_sentiment(df)
        
        print(f"✅ Total news articles: {len(df)}")
        print(f"📊 Mapped to {df['symbol'].nunique()} unique stocks")
        
        return df
    
    def map_news_to_stocks(self, df):
        """Map news articles to stock symbols"""
        print("🔗 Mapping news to stocks...")
        
        mapped_news = []
        
        for _, row in df.iterrows():
            text = f"{row['title']} {row['description']}".lower()
            matched_stocks = []
            
            # Try to match company names
            for symbol, company_name in self.company_names.items():
                if company_name.lower() in text:
                    matched_stocks.append(symbol)
            
            # If no match, try general stock keywords
            if not matched_stocks:
                # Check for sector keywords
                if any(word in text for word in ['bank', 'banking']):
                    matched_stocks.extend([s for s in POPULAR_INDIAN_STOCKS if 'BANK' in s or s in ['SBIN.NS', 'PNB.NS', 'CANBK.NS']])
                
                if any(word in text for word in ['renewable', 'wind', 'solar', 'green energy']):
                    matched_stocks.extend([s for s in POPULAR_INDIAN_STOCKS if s in ['INOXWIND.NS', 'SUZLON.NS', 'ADANIGREEN.NS']])
                
                if any(word in text for word in ['pharma', 'drug', 'medicine']):
                    matched_stocks.extend([s for s in POPULAR_INDIAN_STOCKS if 'PHARMA' in INDIAN_STOCKS.get(s, '')])
                
                if any(word in text for word in ['it sector', 'software', 'technology']):
                    matched_stocks.extend(['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'])
            
            # Create entry for each matched stock
            if matched_stocks:
                for stock in list(set(matched_stocks))[:5]:  # Max 5 stocks per article
                    mapped_news.append({
                        'symbol': stock,
                        'title': row['title'],
                        'description': row['description'],
                        'url': row['url'],
                        'publishedAt': row['publishedAt'],
                        'source': row['source'],
                        'sector': INDIAN_STOCKS.get(stock, 'Others')
                    })
            else:
                # Generic market news - assign to random popular stock
                import random
                stock = random.choice(POPULAR_INDIAN_STOCKS[:50])
                mapped_news.append({
                    'symbol': stock,
                    'title': row['title'],
                    'description': row['description'],
                    'url': row['url'],
                    'publishedAt': row['publishedAt'],
                    'source': row['source'],
                    'sector': INDIAN_STOCKS.get(stock, 'Others')
                })
        
        return pd.DataFrame(mapped_news)
    
    def analyze_sentiment(self, df):
        """Analyze sentiment of news"""
        print("🧠 Analyzing sentiment...")
        
        sentiments = []
        
        for _, row in df.iterrows():
            text = f"{row['title']} {row['description']}".lower()
            
            # Simple keyword-based sentiment
            positive_words = [
                'growth', 'profit', 'gain', 'rise', 'surge', 'up', 'high', 'strong',
                'beat', 'exceed', 'positive', 'upgrade', 'buy', 'bullish', 'rally',
                'expansion', 'contract', 'dividend', 'bonus', 'acquisition', 'merger'
            ]
            
            negative_words = [
                'loss', 'fall', 'drop', 'down', 'decline', 'weak', 'miss', 'cut',
                'negative', 'downgrade', 'sell', 'bearish', 'crash', 'plunge',
                'concern', 'issue', 'problem', 'risk', 'debt', 'investigation'
            ]
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiment = 0.6 + (pos_count * 0.1)
                sentiment = min(sentiment, 0.95)
            elif neg_count > pos_count:
                sentiment = 0.4 - (neg_count * 0.1)
                sentiment = max(sentiment, 0.05)
            else:
                sentiment = 0.5
            
            sentiments.append(sentiment)
        
        df['sentiment_score'] = sentiments
        df['timestamp'] = pd.to_datetime(df['publishedAt'])
        
        return df
    
    def generate_demo_news(self, num_articles=200):
        """Generate demo news if API keys not available"""
        print("📋 Generating demo news...")
        
        import random
        
        news_templates = [
            "{company} reports strong quarterly earnings",
            "{company} announces expansion plans",
            "{company} secures major contract",
            "{company} faces regulatory scrutiny",
            "{company} stock hits new high",
            "{company} declares dividend of ₹{amount} per share",
            "{company} revenue grows by {percent}%",
            "{company} launches new product line"
        ]
        
        news = []
        for _ in range(num_articles):
            symbol = random.choice(POPULAR_INDIAN_STOCKS)
            company = symbol.replace('.NS', '')
            template = random.choice(news_templates)
            
            headline = template.format(
                company=company,
                amount=random.randint(5, 50),
                percent=random.randint(10, 50)
            )
            
            sentiment = random.uniform(0.3, 0.9)
            
            news.append({
                'symbol': symbol,
                'title': headline,
                'description': f"Demo news for {company}",
                'url': 'https://demo.com',
                'publishedAt': datetime.now(self.ist).isoformat(),
                'source': 'Demo Source',
                'sector': INDIAN_STOCKS.get(symbol, 'Others'),
                'sentiment_score': sentiment,
                'timestamp': datetime.now(self.ist)
            })
        
        return pd.DataFrame(news)


# Test function
if __name__ == "__main__":
    fetcher = NewsAPIFetcher()
    news_df = fetcher.fetch_indian_stock_news(max_articles=200)
    print(f"\n✅ Fetched {len(news_df)} news articles")
    print(f"📊 Sample news:\n{news_df.head()}")
    