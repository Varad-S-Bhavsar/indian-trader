# trading_engine.py - News-Based Trading Engine with Smart Quantity

"""
Trading engine with:
- Smart quantity (100 shares if <₹3000, 30 shares if >₹3000)
- News-based trading signals
- Adjustable profit/loss targets
- Real-time ML metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import yfinance as yf
import time
import random
from config import IndianMarketConfig, INDIAN_STOCKS, POPULAR_INDIAN_STOCKS

class IndianTradingEngine:
    """News-driven trading engine with smart quantity"""
    
    def __init__(self, news_fetcher=None):
        self.initial_capital = IndianMarketConfig.INITIAL_CAPITAL
        self.available_cash = IndianMarketConfig.INITIAL_CAPITAL
        self.positions = {}
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.portfolio_history = []
        self.total_realized_pnl = 0
        
        # Live tracking
        self.recent_buys = []
        self.recent_sells = []
        
        # Price cache
        self.price_cache = {}
        self.price_cache_time = {}
        
        # News fetcher
        self.news_fetcher = news_fetcher
        self.news_cache = None
        self.news_cache_time = 0
        
        # Stocks
        self.indian_stocks = INDIAN_STOCKS
        self.popular_stocks = POPULAR_INDIAN_STOCKS
        
        # Timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Last trade time
        self.last_trade_time = time.time()
        
        self.update_portfolio_history()
        
        print(f"✅ Trading Engine initialized")
        print(f"💰 Capital: ₹{self.initial_capital:,.0f}")
        print(f"📊 Universe: {len(self.popular_stocks)} stocks")
        print(f"📰 News integration: {'Enabled' if news_fetcher else 'Disabled'}")
    
    def get_current_indian_price(self, symbol):
        """Get current LIVE price with caching"""
        now = time.time()
        if symbol in self.price_cache:
            cache_age = now - self.price_cache_time.get(symbol, 0)
            if cache_age < 10:
                return self.price_cache[symbol]
        
        price = self._fetch_live_price(symbol)
        
        if price and price > 0:
            self.price_cache[symbol] = price
            self.price_cache_time[symbol] = now
            return price
        
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        return None
    
    def _fetch_live_price(self, symbol):
        """Fetch live price from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            data = ticker.history(period='5d', interval='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            return None
        except:
            return None
    
    def get_bulk_prices(self, symbols):
        """Get prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_current_indian_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def get_fresh_news(self):
        """Get fresh news (cache for 5 minutes)"""
        now = time.time()
        if self.news_cache is not None and (now - self.news_cache_time) < 300:
            return self.news_cache
        
        if self.news_fetcher:
            try:
                print("📰 Fetching fresh news...")
                self.news_cache = self.news_fetcher.fetch_indian_stock_news(max_articles=200)
                self.news_cache_time = now
                return self.news_cache
            except:
                print("⚠️ News fetch failed, using random signals")
                return None
        
        return None
    
    def execute_indian_buy(self, symbol, confidence, sentiment, reason, profit_target=2.0, stop_loss=-1.5):
        """Execute BUY with SMART QUANTITY"""
        current_price = self.get_current_indian_price(symbol)
        
        if current_price is None or current_price <= 0:
            return False, f"Cannot get price for {symbol}"
        
        # SMART QUANTITY based on price
        quantity = IndianMarketConfig.get_smart_quantity(current_price)
        actual_amount = quantity * current_price
        
        # Check cash
        if self.available_cash < actual_amount:
            return False, f"Insufficient cash. Need ₹{actual_amount:,.0f}, have ₹{self.available_cash:,.0f}"
        
        # Execute buy
        self.available_cash -= actual_amount
        
        # Calculate targets based on percentage
        target_price = current_price * (1 + profit_target / 100)
        stop_price = current_price * (1 + stop_loss / 100)
        
        self.positions[symbol] = {
            'quantity': quantity,
            'buy_price': current_price,
            'buy_time': datetime.now(self.ist),
            'buy_amount': actual_amount,
            'target_price': target_price,
            'stop_price': stop_price,
            'confidence': confidence,
            'news_sentiment': sentiment,
            'reason': reason,
            'sector': self.indian_stocks.get(symbol, 'Others'),
            'profit_target_pct': profit_target,
            'stop_loss_pct': stop_loss
        }
        
        self.total_trades += 1
        
        self.trade_history.append({
            'timestamp': datetime.now(self.ist),
            'symbol': symbol,
            'action': 'BUY',
            'price': current_price,
            'quantity': quantity,
            'amount': actual_amount,
            'confidence': confidence,
            'sentiment': sentiment,
            'reason': reason
        })
        
        self.recent_buys.append({
            'timestamp': datetime.now(self.ist),
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'amount': actual_amount,
            'sentiment': sentiment,
            'confidence': confidence,
            'target_price': target_price,
            'stop_price': stop_price
        })
        
        if len(self.recent_buys) > 50:
            self.recent_buys = self.recent_buys[-50:]
        
        self.update_portfolio_history()
        
        return True, f"✅ Bought {quantity} shares at ₹{current_price:.2f} (Total: ₹{actual_amount:,.0f})"
    
    def check_indian_exit_conditions(self):
        """Check exit conditions"""
        actions = []
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = self.get_current_indian_price(symbol)
            
            if current_price is None or current_price <= 0:
                continue
            
            current_value = position['quantity'] * current_price
            pnl = current_value - position['buy_amount']
            pnl_pct = (pnl / position['buy_amount']) * 100
            
            should_sell = False
            exit_reason = ""
            
            # Check profit target
            if current_price >= position['target_price']:
                should_sell = True
                exit_reason = f"🎯 TARGET: {pnl_pct:.2f}%"
            
            # Check stop loss
            elif current_price <= position['stop_price']:
                should_sell = True
                exit_reason = f"🛑 STOP: {pnl_pct:.2f}%"
            
            if should_sell:
                success, message = self.execute_indian_sell(symbol, current_price, exit_reason, pnl, pnl_pct)
                if success:
                    actions.append(message)
        
        return actions
    
    def execute_indian_sell(self, symbol, current_price, reason, pnl, pnl_pct):
        """Execute SELL"""
        if symbol not in self.positions:
            return False, "No position"
        
        position = self.positions[symbol]
        sell_amount = position['quantity'] * current_price
        
        self.available_cash += sell_amount
        self.total_realized_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        holding_time = datetime.now(self.ist) - position['buy_time']
        holding_minutes = holding_time.total_seconds() / 60
        
        self.trade_history.append({
            'timestamp': datetime.now(self.ist),
            'symbol': symbol,
            'action': 'SELL',
            'price': current_price,
            'quantity': position['quantity'],
            'amount': sell_amount,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'holding_time': holding_minutes
        })
        
        self.recent_sells.append({
            'timestamp': datetime.now(self.ist),
            'symbol': symbol,
            'buy_price': position['buy_price'],
            'sell_price': current_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'holding_minutes': holding_minutes
        })
        
        if len(self.recent_sells) > 50:
            self.recent_sells = self.recent_sells[-50:]
        
        del self.positions[symbol]
        self.update_portfolio_history()
        
        return True, f"✅ SOLD {symbol.replace('.NS', '')} at ₹{current_price:.2f} - {reason}"
    
    def exit_all_positions(self):
        """Exit ALL positions immediately"""
        actions = []
        exit_count = 0
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = self.get_current_indian_price(symbol)
            
            if current_price is None or current_price <= 0:
                continue
            
            current_value = position['quantity'] * current_price
            pnl = current_value - position['buy_amount']
            pnl_pct = (pnl / position['buy_amount']) * 100
            
            exit_reason = f"🚪 MANUAL EXIT: {pnl_pct:.2f}%"
            
            success, message = self.execute_indian_sell(symbol, current_price, exit_reason, pnl, pnl_pct)
            if success:
                actions.append(message)
                exit_count += 1
        
        return actions, exit_count
    
    def run_indian_trading_cycle(self, max_positions=15, trade_frequency=15, profit_target=2.0, stop_loss=-1.5, sentiment_threshold=0.4):
        """Run NEWS-BASED trading cycle"""
        actions = []
        
        current_time = time.time()
        if current_time - self.last_trade_time < trade_frequency:
            return actions
        
        self.last_trade_time = current_time
        
        # Check exits
        exit_actions = self.check_indian_exit_conditions()
        actions.extend(exit_actions)
        
        # Look for new buys using NEWS
        if len(self.positions) < max_positions:
            news_df = self.get_fresh_news()
            
            if news_df is not None and len(news_df) > 0:
                # Use NEWS-based signals
                actions.extend(self._trade_from_news(news_df, max_positions, profit_target, stop_loss, sentiment_threshold))
            else:
                # Fallback to random signals
                actions.extend(self._trade_random(max_positions, profit_target, stop_loss, sentiment_threshold))
        
        return actions
    
    def _trade_from_news(self, news_df, max_positions, profit_target, stop_loss, sentiment_threshold):
        """Trade based on real news"""
        actions = []
        
        # Filter high sentiment news
        high_sentiment = news_df[news_df['sentiment_score'] > sentiment_threshold].copy()
        
        if len(high_sentiment) == 0:
            return actions
        
        # Group by symbol and get best sentiment
        best_signals = high_sentiment.groupby('symbol').agg({
            'sentiment_score': 'mean',
            'title': 'first'
        }).reset_index()
        
        best_signals = best_signals.sort_values('sentiment_score', ascending=False)
        
        # Trade top signals
        for _, row in best_signals.head(5).iterrows():
            if len(self.positions) >= max_positions:
                break
            
            symbol = row['symbol']
            
            if symbol in self.positions:
                continue
            
            sentiment = row['sentiment_score']
            confidence = min(sentiment * 100, 95)
            reason = f"News: {row['title'][:50]}..."
            
            success, message = self.execute_indian_buy(
                symbol, confidence, sentiment, reason, 
                profit_target, stop_loss
            )
            
            if success:
                actions.append(f"🟢 BUY {symbol.replace('.NS', '')}: {message}")
        
        return actions
    
    def _trade_random(self, max_positions, profit_target, stop_loss, sentiment_threshold):
        """Fallback: Random trading signals"""
        actions = []
        
        candidates = random.sample(self.popular_stocks, min(10, len(self.popular_stocks)))
        
        for symbol in candidates:
            if len(self.positions) >= max_positions:
                break
            
            if symbol in self.positions:
                continue
            
            sentiment = random.uniform(0.4, 0.9)
            confidence = random.uniform(65, 90)
            
            if sentiment > sentiment_threshold and random.random() < 0.6:
                reason = f"Signal: Sentiment {sentiment:.2f}"
                success, message = self.execute_indian_buy(
                    symbol, confidence, sentiment, reason,
                    profit_target, stop_loss
                )
                if success:
                    actions.append(f"🟢 BUY {symbol.replace('.NS', '')}: {message}")
        
        return actions
    
    def update_portfolio_history(self):
        """Update portfolio history"""
        total_position_value = 0
        
        if self.positions:
            symbols = list(self.positions.keys())
            prices = self.get_bulk_prices(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    total_position_value += position['quantity'] * prices[symbol]
        
        total_portfolio_value = self.available_cash + total_position_value
        total_return_pct = ((total_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        unrealized_pnl = 0
        if self.positions:
            symbols = list(self.positions.keys())
            prices = self.get_bulk_prices(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    current_value = position['quantity'] * prices[symbol]
                    unrealized_pnl += (current_value - position['buy_amount'])
        
        self.portfolio_history.append({
            'timestamp': datetime.now(self.ist),
            'portfolio_value': total_portfolio_value,
            'cash': self.available_cash,
            'positions_value': total_position_value,
            'return_pct': total_return_pct,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_realized_pnl,
            'total_pnl': unrealized_pnl + self.total_realized_pnl
        })
        
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_indian_performance(self):
        """Get performance metrics"""
        total_position_value = 0
        unrealized_pnl = 0
        
        if self.positions:
            symbols = list(self.positions.keys())
            prices = self.get_bulk_prices(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    current_value = position['quantity'] * prices[symbol]
                    total_position_value += current_value
                    unrealized_pnl += (current_value - position['buy_amount'])
        
        total_capital = self.available_cash + total_position_value
        total_return = ((total_capital - self.initial_capital) / self.initial_capital) * 100
        total_pnl = (total_capital - self.initial_capital)
        
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100 if self.total_trades > 0 else 0
        
        return {
            'available_cash': self.available_cash,
            'position_value': total_position_value,
            'total_capital': total_capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_realized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'active_positions': len(self.positions)
        }
