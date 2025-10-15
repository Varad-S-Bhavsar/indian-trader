# Portfolio Manager - $5 Million Virtual Capital Management
"""
Advanced portfolio management with risk controls and performance tracking
Handles $5 million virtual capital with real-time P&L monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import yfinance as yf  # For live US price fetching
sys.path.append('..')
import config

class PortfolioManager:
    """Manages $5 million virtual trading portfolio"""
    
    def __init__(self, initial_capital=config.VIRTUAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_cash = initial_capital
        
        # Holdings: {symbol: {quantity, avg_price, invested_amount, current_price, pnl}}
        self.holdings = {}
        
        # Trade history
        self.trade_history = []
        
        # Performance metrics
        self.daily_pnl = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Risk management
        self.max_position_size = config.MAX_POSITION_SIZE * initial_capital
        self.profit_target = config.PROFIT_TARGET  # 2%
        self.stop_loss = config.STOP_LOSS  # -2%
        
        print(f"💰 Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def get_live_price(self, symbol):
        """Fetch live US stock price using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")  # 1-minute interval for near-real-time
            if not data.empty:
                return round(data['Close'].iloc[-1], 2)
            else:
                print(f"No data available for {symbol}")
                return None
        except Exception as e:
            print(f"Error fetching price for {symbol}: {str(e)}")
            # Fallback to random simulation if API fails
            base_price = 100  # Adjust based on stock
            return round(base_price * np.random.uniform(0.97, 1.03), 2)
    
    def calculate_position_size(self, symbol, signal_confidence):
        """Calculate optimal position size based on confidence and risk"""
        
        # Base allocation based on confidence
        base_allocation = min(signal_confidence * 0.1, 0.08)  # Max 8% per stock
        
        # Adjust based on portfolio concentration
        current_positions = len(self.holdings)
        if current_positions > 10:
            base_allocation *= 0.8  # Reduce size if too many positions
        
        # Calculate investment amount
        investment_amount = self.available_cash * base_allocation
        investment_amount = min(investment_amount, self.max_position_size)
        
        return investment_amount
    
    def buy_stock(self, symbol, signal_confidence, current_price=None, reason="ML Signal"):
        """Execute buy order with risk management"""
        
        if current_price is None:
            current_price = self.get_live_price(symbol)
        
        # Calculate position size
        investment_amount = self.calculate_position_size(symbol, signal_confidence)
        
        if investment_amount < 1000:  # Minimum $1000 investment
            return {
                'success': False,
                'message': f"Investment amount too small: ${investment_amount:.2f}"
            }
        
        if investment_amount > self.available_cash:
            return {
                'success': False, 
                'message': f"Insufficient funds. Available: ${self.available_cash:.2f}"
            }
        
        # Calculate quantity
        quantity = int(investment_amount / current_price)
        actual_investment = quantity * current_price
        
        if quantity == 0:
            return {
                'success': False,
                'message': f"Cannot afford even 1 share at ${current_price:.2f}"
            }
        
        # Execute trade
        if symbol in self.holdings:
            # Add to existing position
            old_qty = self.holdings[symbol]['quantity']
            old_invested = self.holdings[symbol]['invested_amount']
            
            new_qty = old_qty + quantity
            new_invested = old_invested + actual_investment
            new_avg_price = new_invested / new_qty
            
            self.holdings[symbol].update({
                'quantity': new_qty,
                'avg_price': new_avg_price,
                'invested_amount': new_invested
            })
        else:
            # New position
            self.holdings[symbol] = {
                'quantity': quantity,
                'avg_price': current_price,
                'invested_amount': actual_investment,
                'entry_time': datetime.now(),
                'target_price': current_price * (1 + self.profit_target/100),
                'stop_loss_price': current_price * (1 + self.stop_loss/100)
            }
        
        # Update cash
        self.available_cash -= actual_investment
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': current_price,
            'amount': actual_investment,
            'reason': reason,
            'confidence': signal_confidence
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
        
        return {
            'success': True,
            'message': f"Bought {quantity} shares of {symbol} at ${current_price:.2f}",
            'quantity': quantity,
            'price': current_price,
            'amount': actual_investment
        }
    
    def sell_stock(self, symbol, quantity=None, current_price=None, reason="Take Profit"):
        """Execute sell order"""
        
        if symbol not in self.holdings:
            return {'success': False, 'message': f"No holdings in {symbol}"}
        
        if current_price is None:
            current_price = self.get_live_price(symbol)
        
        holding = self.holdings[symbol]
        available_qty = holding['quantity']
        
        # Default to selling all shares
        if quantity is None or quantity > available_qty:
            quantity = available_qty
        
        # Calculate sale proceeds
        sale_amount = quantity * current_price
        
        # Calculate P&L
        avg_price = holding['avg_price']
        pnl = (current_price - avg_price) * quantity
        pnl_percentage = ((current_price / avg_price) - 1) * 100
        
        # Update holdings
        if quantity == available_qty:
            # Selling entire position
            del self.holdings[symbol]
        else:
            # Partial sale
            remaining_qty = available_qty - quantity
            remaining_invested = avg_price * remaining_qty
            
            self.holdings[symbol].update({
                'quantity': remaining_qty,
                'invested_amount': remaining_invested
            })
        
        # Update cash
        self.available_cash += sale_amount
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': current_price,
            'amount': sale_amount,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        
        # Update win/loss statistics
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        return {
            'success': True,
            'message': f"Sold {quantity} shares of {symbol} at ${current_price:.2f}",
            'quantity': quantity,
            'price': current_price,
            'amount': sale_amount,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
        }
    
    def check_stop_loss_take_profit(self):
        """Check all positions for stop loss and take profit triggers"""
        triggers = []
        
        for symbol, holding in list(self.holdings.items()):
            current_price = self.get_live_price(symbol)
            entry_price = holding['avg_price']
            
            # Calculate current return
            current_return = ((current_price / entry_price) - 1) * 100
            
            # Check take profit (2% target)
            if current_return >= self.profit_target:
                result = self.sell_stock(symbol, current_price=current_price, 
                                       reason="Take Profit - 2% Target Hit")
                triggers.append({
                    'symbol': symbol,
                    'action': 'TAKE_PROFIT',
                    'return': current_return,
                    'result': result
                })
            
            # Check stop loss (-2% limit)
            elif current_return <= self.stop_loss:
                result = self.sell_stock(symbol, current_price=current_price,
                                       reason="Stop Loss - 2% Limit Hit") 
                triggers.append({
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'return': current_return,
                    'result': result
                })
        
        return triggers
    
    def get_portfolio_summary(self):
        """Get comprehensive portfolio summary"""
        total_invested = 0
        total_current_value = 0
        positions = []
        
        for symbol, holding in self.holdings.items():
            current_price = self.get_live_price(symbol)
            invested = holding['invested_amount']
            current_value = holding['quantity'] * current_price
            pnl = current_value - invested
            pnl_pct = ((current_value / invested) - 1) * 100
            
            total_invested += invested
            total_current_value += current_value
            
            positions.append({
                'symbol': symbol,
                'quantity': holding['quantity'],
                'avg_price': holding['avg_price'],
                'current_price': current_price,
                'invested': invested,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percentage': pnl_pct,
                'weight': (current_value / (total_current_value or 1)) * 100
            })
        
        # Overall portfolio metrics
        total_portfolio_value = self.available_cash + total_current_value
        total_pnl = total_portfolio_value - self.initial_capital
        total_return = ((total_portfolio_value / self.initial_capital) - 1) * 100
        
        # Performance metrics
        win_rate = (self.winning_trades / max(self.winning_trades + self.losing_trades, 1)) * 100
        
        summary = {
            'initial_capital': self.initial_capital,
            'available_cash': self.available_cash,
            'invested_amount': total_invested,
            'portfolio_value': total_current_value,
            'total_portfolio_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'positions': positions,
            'num_positions': len(positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate
        }
        
        return summary
    
    def get_trade_history(self, limit=50):
        """Get recent trade history"""
        return sorted(self.trade_history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def save_portfolio_state(self, filepath="data/portfolio_state.json"):
        """Save portfolio state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'available_cash': self.available_cash,
            'holdings': self.holdings,
            'trade_history': [
                {**trade, 'timestamp': trade['timestamp'].isoformat()} 
                for trade in self.trade_history
            ],
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"💾 Portfolio state saved to {filepath}")
    
    def load_portfolio_state(self, filepath="data/portfolio_state.json"):
        """Load portfolio state from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.initial_capital = state['initial_capital']
            self.current_capital = state['current_capital'] 
            self.available_cash = state['available_cash']
            self.holdings = state['holdings']
            self.total_trades = state['performance']['total_trades']
            self.winning_trades = state['performance']['winning_trades']
            self.losing_trades = state['performance']['losing_trades']
            
            # Restore trade history
            self.trade_history = []
            for trade in state['trade_history']:
                trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                self.trade_history.append(trade)
            
            print(f"📂 Portfolio state loaded from {filepath}")
            return True
        
        return False

# Test the portfolio manager
if __name__ == "__main__":
    portfolio = PortfolioManager()
    
    print("\n🧪 Testing Portfolio Manager:")
    print("="*50)
    
    # Test buying stocks
    result1 = portfolio.buy_stock('AAPL', 0.85, reason="Strong Sentiment + ML")
    print(f"Buy AAPL: {result1['message']}")
    
    result2 = portfolio.buy_stock('MSFT', 0.78, reason="Positive Earnings")  
    print(f"Buy MSFT: {result2['message']}")
    
    # Check portfolio
    summary = portfolio.get_portfolio_summary()
    print(f"\n📊 Portfolio Summary:")
    print(f"Total Value: ${summary['total_portfolio_value']:,.0f}")
    print(f"P&L: ${summary['total_pnl']:,.0f} ({summary['total_return_pct']:.2f}%)")
    print(f"Positions: {summary['num_positions']}")
    
    print("\n✅ Portfolio Manager Ready!")
