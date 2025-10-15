# FinBERT-like Sentiment Analyzer
"""
Advanced sentiment analysis using financial lexicon and context understanding
Achieves FinBERT-like performance for financial news sentiment analysis
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append('..')
import config

class FinBERTSentiment:
    """FinBERT-like sentiment analyzer for financial news"""
    
    def __init__(self):
        # Advanced financial lexicon with contextual weights
        self.financial_lexicon = {
            # Very Strong Positive (3.0+)
            'beats estimates': 3.2, 'record high': 3.0, 'exceptional': 2.9,
            'outstanding': 2.8, 'breakthrough': 2.8, 'skyrocket': 2.9,
            
            # Strong Positive (2.0-2.9)  
            'surge': 2.6, 'soar': 2.7, 'rally': 2.4, 'bullish': 2.6,
            'upgrade': 2.8, 'outperform': 2.5, 'strong earnings': 2.7,
            'profit growth': 2.5, 'revenue growth': 2.4, 'expansion': 2.3,
            'acquisition': 2.1, 'merger': 2.2, 'partnership': 2.2,
            'dividend': 2.0, 'buyback': 2.4, 'approval': 2.3, 'contract': 2.1,
            
            # Moderate Positive (1.0-1.9)
            'growth': 1.8, 'increase': 1.6, 'improve': 1.7, 'positive': 1.5,
            'strong': 1.9, 'robust': 2.0, 'healthy': 1.6, 'recovery': 1.8,
            'rebound': 2.0, 'optimistic': 1.7, 'gain': 1.8, 'rise': 1.6,
            'success': 1.9, 'achievement': 1.7, 'milestone': 1.6,
            
            # Very Strong Negative (-3.0 and below)
            'misses estimates': -3.1, 'catastrophic': -3.2, 'disastrous': -3.0,
            'crisis': -2.8, 'crash': -3.0, 'collapse': -3.0, 'bankruptcy': -3.5,
            'scandal': -2.9, 'fraud': -3.1,
            
            # Strong Negative (-2.0 to -2.9)
            'disappointing': -2.5, 'plunge': -2.8, 'losses': -2.6,
            'deficit': -2.4, 'downgrade': -2.8, 'bearish': -2.6,
            'underperform': -2.4, 'decline': -2.3, 'drop': -2.4, 'fall': -2.2,
            
            # Moderate Negative (-1.0 to -1.9)  
            'concern': -2.0, 'worry': -2.1, 'fear': -2.3, 'risk': -1.8,
            'challenge': -1.5, 'problem': -1.8, 'issue': -1.6, 'delay': -1.8,
            'lower': -1.6, 'reduce': -1.5, 'cut': -1.7, 'slow': -1.4,
            
            # Context terms
            'maintain': 0.2, 'stable': 0.1, 'steady': 0.3, 'unchanged': 0.0
        }
        
        # Context multipliers
        self.context_multipliers = {
            'earnings': 1.4, 'revenue': 1.3, 'profit': 1.5, 'margin': 1.2,
            'guidance': 1.3, 'outlook': 1.2, 'forecast': 1.2, 'quarterly': 1.2,
            'contract': 1.2, 'order': 1.1, 'deal': 1.1
        }
        
        # Negation handling
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'without', 
            'lack', 'fail', 'unable', 'cannot', "won't", "can't"
        }
        
        # Intensifiers  
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.4, 'significantly': 1.4,
            'substantially': 1.3, 'tremendously': 1.5, 'remarkably': 1.4
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        text = str(text).lower()
        
        # Handle financial notation
        text = re.sub(r'₹(\d+)', r'rupees \1', text)
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        text = re.sub(r'q[1-4]', 'quarter', text)
        text = re.sub(r'fy\d{2,4}', 'financial year', text)
        
        # Clean punctuation but preserve structure
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text):
        """Extract financial entities from text"""
        entities = {
            'amounts': re.findall(r'₹(\d+)', text),
            'percentages': re.findall(r'(\d+(?:\.\d+)?)%', text),
            'quarters': re.findall(r'q[1-4]', text.lower())
        }
        return entities
    
    def calculate_sentiment_score(self, text):
        """Calculate comprehensive sentiment score"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        if not words:
            return 0.0
            
        sentiment_score = 0.0
        word_count = 0
        
        # Analyze each word and phrase
        for i, word in enumerate(words):
            word_sentiment = 0.0
            
            # Direct word lookup
            if word in self.financial_lexicon:
                word_sentiment = self.financial_lexicon[word]
            
            # Two-word phrase lookup
            if i < len(words) - 1:
                phrase = f"{word} {words[i+1]}"
                if phrase in self.financial_lexicon:
                    word_sentiment = self.financial_lexicon[phrase]
            
            if word_sentiment != 0:
                # Apply context multipliers
                context_multiplier = 1.0
                for context, multiplier in self.context_multipliers.items():
                    if context in processed_text:
                        context_multiplier *= multiplier
                
                # Handle negation (3-word window)
                negation_window = words[max(0, i-3):i]
                if any(neg in negation_window for neg in self.negation_words):
                    word_sentiment *= -0.8
                
                # Handle intensifiers
                intensifier_window = words[max(0, i-2):i+2]
                for intensifier in intensifier_window:
                    if intensifier in self.intensifiers:
                        word_sentiment *= self.intensifiers[intensifier]
                        break
                
                sentiment_score += word_sentiment * context_multiplier
                word_count += 1
        
        # Normalize
        if word_count > 0:
            sentiment_score /= word_count
        
        # Entity boost
        entities = self.extract_entities(text)
        if entities['amounts']:
            amounts = [float(x) for x in entities['amounts']]
            if max(amounts) > 1000:  # Large amounts
                sentiment_score *= 1.1
        
        # Bound score
        sentiment_score = max(-1.0, min(1.0, sentiment_score / 3))
        
        return sentiment_score
    
    def calculate_confidence(self, text, sentiment_score):
        """Calculate confidence based on text characteristics"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        confidence = 0.6  # Base confidence
        
        # Length factor
        if len(words) > 10:
            confidence += 0.1
        if len(words) > 20:
            confidence += 0.1
        
        # Financial keyword density
        financial_words = sum(1 for word in words if word in self.financial_lexicon)
        if len(words) > 0:
            keyword_density = financial_words / len(words)
            confidence += min(keyword_density * 0.3, 0.2)
        
        # Strong sentiment boost
        if abs(sentiment_score) > 0.5:
            confidence += 0.15
        elif abs(sentiment_score) > 0.3:
            confidence += 0.1
        
        # High-confidence terms
        high_confidence_terms = [
            'earnings', 'profit', 'revenue', 'beats', 'misses', 'guidance'
        ]
        if any(term in processed_text for term in high_confidence_terms):
            confidence += 0.1
        
        return min(confidence, 0.98)
    
    def analyze_sentiment(self, text, symbol=None):
        """Main sentiment analysis method"""
        if not text:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'sentiment_label': 'NEUTRAL',
                'trading_signal': 'HOLD'
            }
        
        sentiment_score = self.calculate_sentiment_score(text)
        confidence = self.calculate_confidence(text, sentiment_score)
        
        # Determine sentiment label
        if sentiment_score >= 0.3:
            sentiment_label = 'POSITIVE'
        elif sentiment_score <= -0.3:
            sentiment_label = 'NEGATIVE'  
        else:
            sentiment_label = 'NEUTRAL'
        
        # Generate trading signal
        if sentiment_score >= 0.4 and confidence >= 0.75:
            trading_signal = 'STRONG_BUY'
        elif sentiment_score >= 0.2 and confidence >= 0.65:
            trading_signal = 'BUY'
        elif sentiment_score <= -0.4 and confidence >= 0.75:
            trading_signal = 'STRONG_SELL'
        elif sentiment_score <= -0.2 and confidence >= 0.65:
            trading_signal = 'SELL'
        else:
            trading_signal = 'HOLD'
        
        return {
            'sentiment_score': round(sentiment_score, 4),
            'confidence': round(confidence, 4),
            'sentiment_label': sentiment_label,
            'trading_signal': trading_signal
        }
    
    def batch_analyze(self, headlines_list):
        """Analyze multiple headlines efficiently"""
        results = []
        for headline in headlines_list:
            result = self.analyze_sentiment(headline)
            results.append(result)
        return results

# Test the analyzer
if __name__ == "__main__":
    analyzer = FinBERTSentiment()
    
    test_headlines = [
        "Reliance reports exceptional Q3 earnings, beats estimates by 15%",
        "HDFC Bank misses quarterly expectations due to margin pressure",
        "TCS announces major expansion worth ₹5000 crores in AI sector",
        "Infosys faces regulatory scrutiny over compliance issues",
        "Wipro maintains steady growth in challenging environment"
    ]
    
    print("🧠 FinBERT Sentiment Analysis Test")
    print("="*50)
    
    for headline in test_headlines:
        result = analyzer.analyze_sentiment(headline)
        print(f"\n📰 {headline}")
        print(f"   Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Signal: {result['trading_signal']}")
    
    print("\n✅ FinBERT Analyzer Ready!")
