# indian_news_generator.py - Indian Stock Market News Generator

import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional
import logging
from config import INDIAN_STOCKS, POPULAR_INDIAN_STOCKS, IndianMarketConfig

class IndianNewsGenerator:
    """Generate news for Indian stock market"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.stocks = INDIAN_STOCKS
        self.popular_stocks = POPULAR_INDIAN_STOCKS
        
        # Indian market specific news templates
        self.news_templates = {
            'positive': [
                "{company} reports strong Q{quarter} results, beats analyst estimates",
                "{company} announces major expansion plans worth ₹{amount} crores",
                "{company} bags large contract from government, stock rallies",
                "{company} declares bonus shares, shareholders rejoice",
                "{company} announces stock split to make shares more affordable",
                "{company} enters into strategic partnership with global leader",
                "{company} launches innovative product in Indian market",
                "{company} receives regulatory approval for new venture",
                "{company} management provides optimistic guidance for FY{year}",
                "{company} dividend announcement exceeds market expectations",
                "Brokerages upgrade {company} target price citing strong fundamentals",
                "{company} digital transformation shows promising results",
                "FII investment in {company} increases significantly this quarter",
                "{company} emerges as leader in ESG ratings among peers",
                "{company} exports surge, contributing to revenue growth"
            ],
            'negative': [
                "{company} faces regulatory scrutiny over compliance issues",
                "{company} Q{quarter} results disappoint, misses revenue targets",
                "{company} announces workforce restructuring amid cost pressures",
                "{company} facing supply chain disruptions in key markets",
                "{company} management warns of challenging business environment",
                "Regulatory changes impact {company} business model negatively",
                "{company} faces increased competition in core segment",
                "Raw material costs pressure {company} profit margins",
                "{company} delays major project due to funding constraints",
                "Brokerages downgrade {company} on weak demand outlook",
                "FII selling pressure weighs on {company} stock price",
                "{company} litigation costs expected to impact near-term profitability",
                "Currency headwinds affect {company} export revenues",
                "{company} faces environmental compliance challenges",
                "Labor disputes at {company} facilities disrupt operations"
            ],
            'neutral': [
                "{company} management to address investors in quarterly call",
                "{company} board meeting scheduled to discuss expansion plans",
                "{company} announces date for annual general meeting",
                "{company} stock splits approved by board of directors",
                "{company} appoints new independent directors to board",
                "{company} receives ISO certification for quality standards",
                "{company} participates in major industry conference",
                "{company} announces CSR initiatives for rural development",
                "Analysts maintain hold rating on {company} stock",
                "{company} updates on regulatory filing timeline",
                "{company} subsidiary incorporation completed in new state",
                "{company} enters into non-disclosure agreement for potential deal",
                "{company} announces employee stock option plan",
                "{company} launches investor relations portal",
                "{company} releases sustainability report for FY{year}"
            ]
        }
        
        self.company_names = {
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'RELIANCE.NS': 'Reliance Industries',
            'MARUTI.NS': 'Maruti Suzuki',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ITC.NS': 'ITC',
            'LT.NS': 'Larsen & Toubro',
            'AXISBANK.NS': 'Axis Bank',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'TATAMOTORS.NS': 'Tata Motors',
            'WIPRO.NS': 'Wipro',
            'ULTRACEMCO.NS': 'UltraTech Cement'
        }
    
    def generate_sentiment_score(self, news_type: str) -> float:
        """Generate realistic sentiment score for Indian market"""
        if news_type == 'positive':
            return random.uniform(0.4, 0.9)  # Strong positive sentiment
        elif news_type == 'negative':
            return random.uniform(-0.8, -0.2)  # Strong negative sentiment
        else:  # neutral
            return random.uniform(-0.15, 0.15)  # Neutral sentiment
    
    def generate_indian_news(self, num_articles: int = 50) -> pd.DataFrame:
        """Generate Indian stock market news"""
        
        news_data = []
        
        for _ in range(num_articles):
            # Select random stock and news type
            symbol = random.choice(self.popular_stocks)
            news_type = random.choices(
                ['positive', 'negative', 'neutral'],
                weights=[0.4, 0.25, 0.35],  # More positive bias in bull market
                k=1
            )[0]
            
            # Generate news content
            template = random.choice(self.news_templates[news_type])
            company = self.company_names.get(symbol, symbol.replace('.NS', ''))
            
            # Fill template variables
            headline = template.format(
                company=company,
                quarter=random.randint(1, 4),
                amount=random.randint(100, 5000),
                year=random.choice([2024, 2025]),
            )
            
            # Generate metadata
            sentiment_score = self.generate_sentiment_score(news_type)
            timestamp = datetime.now(self.ist) - timedelta(
                hours=random.randint(0, 24),
                minutes=random.randint(0, 59)
            )
            
            news_item = {
                'symbol': symbol,
                'company': company,
                'headline': headline,
                'sentiment_score': sentiment_score,
                'news_type': news_type,
                'timestamp': timestamp,
                'source': random.choice([
                    'Economic Times', 'Business Standard', 'Mint', 'Moneycontrol',
                    'CNBC TV18', 'BloombergQuint', 'The Hindu BusinessLine',
                    'Financial Express', 'Business Today', 'Money Control'
                ]),
                'sector': self.stocks.get(symbol, 'Others'),
                'is_real': False,  # Generated news
                'market_cap': self.get_market_cap_category(symbol),
                'exchange': 'NSE'
            }
            
            news_data.append(news_item)
        
        return pd.DataFrame(news_data)
    
    def get_market_cap_category(self, symbol: str) -> str:
        """Categorize Indian stocks by market cap"""
        large_cap = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'RELIANCE.NS']
        mid_cap = ['MARUTI.NS', 'SUNPHARMA.NS', 'HINDUNILVR.NS', 'ITC.NS']
        
        if symbol in large_cap:
            return 'Large Cap'
        elif symbol in mid_cap:
            return 'Mid Cap'
        else:
            return 'Small Cap'
    
    def get_real_indian_news(self, newsapi_key: str) -> pd.DataFrame:
        """Fetch real news from Indian financial sources"""
        if not newsapi_key:
            return pd.DataFrame()
        
        try:
            # Indian financial news sources
            indian_sources = [
                'the-times-of-india',
                'the-hindu',
                'business-standard'
            ]
            
            url = f"https://newsapi.org/v2/everything"
            
            all_articles = []
            
            # Search for Indian market terms
            indian_queries = [
                'NSE BSE Indian stock market',
                'Sensex Nifty Indian shares',
                'Indian companies quarterly results',
                'RBI monetary policy India',
                'FII DII Indian market'
            ]
            
            for query in indian_queries[:2]:  # Limit to avoid API quota
                params = {
                    'q': query,
                    'apiKey': newsapi_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        # Try to extract Indian stock symbol from title/content
                        symbol = self.extract_indian_symbol(article['title'] + ' ' + (article.get('description', '')))
                        
                        if symbol:
                            processed_article = {
                                'symbol': symbol,
                                'company': self.company_names.get(symbol, symbol.replace('.NS', '')),
                                'headline': article['title'],
                                'description': article.get('description', ''),
                                'url': article['url'],
                                'source': article['source']['name'],
                                'timestamp': pd.to_datetime(article['publishedAt']),
                                'sentiment_score': self.calculate_sentiment_from_text(article['title']),
                                'sector': self.stocks.get(symbol, 'Others'),
                                'is_real': True,
                                'exchange': 'NSE'
                            }
                            
                            all_articles.append(processed_article)
            
            return pd.DataFrame(all_articles)
        
        except Exception as e:
            print(f"Error fetching Indian news: {e}")
            return pd.DataFrame()
    
    def extract_indian_symbol(self, text: str) -> Optional[str]:
        """Extract Indian stock symbol from news text"""
        text_upper = text.upper()
        
        # Company name mappings
        company_mappings = {
            'TCS': 'TCS.NS',
            'TATA CONSULTANCY': 'TCS.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'RELIANCE': 'RELIANCE.NS',
            'MARUTI': 'MARUTI.NS',
            'SUZUKI': 'MARUTI.NS',
            'SUN PHARMA': 'SUNPHARMA.NS',
            'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'L&T': 'LT.NS',
            'LARSEN': 'LT.NS',
            'AXIS BANK': 'AXISBANK.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'TATA MOTORS': 'TATAMOTORS.NS',
            'WIPRO': 'WIPRO.NS'
        }
        
        for company, symbol in company_mappings.items():
            if company in text_upper:
                return symbol
        
        # Fallback to random popular stock for general Indian market news
        if any(term in text_upper for term in ['SENSEX', 'NIFTY', 'NSE', 'BSE', 'INDIAN MARKET']):
            return random.choice(self.popular_stocks)
        
        return None
    
    def calculate_sentiment_from_text(self, text: str) -> float:
        """Simple sentiment calculation for Indian market"""
        positive_words = [
            'surge', 'rally', 'gain', 'rise', 'up', 'bull', 'strong', 'robust',
            'growth', 'profit', 'beat', 'exceed', 'upgrade', 'positive',
            'optimistic', 'expansion', 'launch', 'partnership', 'agreement'
        ]
        
        negative_words = [
            'fall', 'drop', 'decline', 'bear', 'weak', 'loss', 'miss',
            'disappoint', 'concern', 'worry', 'downgrade', 'negative',
            'pressure', 'challenge', 'risk', 'disruption', 'delay'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return random.uniform(0.3, 0.8)
        elif negative_count > positive_count:
            return random.uniform(-0.6, -0.1)
        else:
            return random.uniform(-0.2, 0.2)

class IndianAutoNewsRefresher:
    """Auto-refresh Indian news data"""
    
    def __init__(self, refresh_interval_minutes: int = 30):
        self.refresh_interval = refresh_interval_minutes
        self.news_generator = IndianNewsGenerator()
        self.last_refresh = None
        self.cached_news = pd.DataFrame()
    
    def get_fresh_indian_news_data(self, newsapi_key: str = None) -> pd.DataFrame:
        """Get fresh Indian news data"""
        
        # Check if refresh needed
        if (self.last_refresh is None or 
            (datetime.now() - self.last_refresh).seconds > (self.refresh_interval * 60) or
            self.cached_news.empty):
            
            print("🔄 Refreshing Indian news data...")
            
            # Generate synthetic news
            synthetic_news = self.news_generator.generate_indian_news(40)
            
            # Try to get real news if API key available
            real_news = pd.DataFrame()
            if newsapi_key:
                real_news = self.news_generator.get_real_indian_news(newsapi_key)
            
            # Combine news sources
            if not real_news.empty:
                # Ensure consistent columns
                common_cols = list(set(synthetic_news.columns) & set(real_news.columns))
                synthetic_news = synthetic_news[common_cols]
                real_news = real_news[common_cols]
                
                self.cached_news = pd.concat([real_news, synthetic_news], ignore_index=True)
            else:
                self.cached_news = synthetic_news
            
            # Sort by timestamp and sentiment
            self.cached_news = self.cached_news.sort_values(
                ['timestamp', 'sentiment_score'], 
                ascending=[False, False]
            ).reset_index(drop=True)
            
            self.last_refresh = datetime.now()
            print(f"✅ Loaded {len(self.cached_news)} Indian news articles")
        
        return self.cached_news.copy()
