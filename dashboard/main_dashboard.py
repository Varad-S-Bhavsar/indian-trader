# Complete Fixed Dashboard with Component Loading
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
import os
import pytz

# Add paths
sys.path.append('.')
sys.path.append('./data')
sys.path.append('./trading')
sys.path.append('./models')

import config

# Component loading function
@st.cache_resource
def load_components():
    """Load all trading system components"""
    try:
        from data.synthetic_news_generator import SyntheticNewsGenerator, AutoDataRefresher
        from sentiment_analyzer import SentimentAnalyzer  
        from prediction_model import PredictionModel
        from enhanced_portfolio_manager import PortfolioManager
        from trading_engine import TradingEngine
        
        # Initialize components
        news_generator = SyntheticNewsGenerator()
        sentiment_analyzer = SentimentAnalyzer()
        prediction_model = PredictionModel()
        portfolio_manager = PortfolioManager()
        trading_engine = TradingEngine(portfolio_manager, sentiment_analyzer, prediction_model)
        
        return news_generator, sentiment_analyzer, prediction_model, portfolio_manager, trading_engine
    except Exception as e:
        st.error(f"Error importing components: {str(e)}")
        return None

@st.cache_data(ttl=120, show_spinner=False)  # Cache for 2 minutes only
def load_news_data_enhanced(force_refresh=False, newsapi_key=""):
    """Load enhanced news data with NewsAPI + synthetic mix"""
    
    # Initialize with NewsAPI key from config or user input
    api_key = newsapi_key or config.NEWSAPI_KEY
    
    from data.synthetic_news_generator import SyntheticNewsGenerator, AutoDataRefresher
    
    refresher = AutoDataRefresher(refresh_interval_minutes=config.AUTO_REFRESH_INTERVAL)
    generator = SyntheticNewsGenerator(newsapi_key=api_key if api_key else None)
    
    if force_refresh:
        refresher.force_refresh_data()
        st.success("🔄 Forced refresh - new data generated!")
    
    try:
        return refresher.get_fresh_data(generator)
    except Exception as e:
        st.error(f"Error loading news: {str(e)}")
        # Fallback to synthetic only
        generator_fallback = SyntheticNewsGenerator()
        return generator_fallback.generate_mixed_news_dataset(200)

def debug_and_fix_trading():
    """Complete trading debug and manual execution"""
    
    st.subheader("🔍 Trading System Debug & Manual Controls")
    
    # Check market hours
    now_et = datetime.now(pytz.timezone('America/New_York'))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = market_open.time() <= now_et.time() <= market_close.time() and now_et.weekday() < 5
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🕐 Current ET Time", now_et.strftime('%H:%M:%S'))
    with col2:
        st.metric("📊 Market Status", "🟢 OPEN" if is_open else "🔴 CLOSED")
    with col3:
        weekday = "Weekday" if now_et.weekday() < 5 else "Weekend"
        st.metric("📅 Day Type", weekday)
    
    # System Status Checks
    st.subheader("🔧 System Status")
    
    # Check trading engine
    if 'trading_engine' in st.session_state:
        engine = st.session_state.trading_engine
        engine_status = "🟢 ACTIVE" if engine.trading_active else "🔴 STOPPED"
        st.write(f"**Trading Engine:** {engine_status}")
        
        if not engine.trading_active:
            if st.button("▶️ Start Trading Engine"):
                engine.start_trading()
                st.success("Trading engine started!")
                st.experimental_rerun()
    else:
        st.error("❌ Trading Engine: NOT INITIALIZED")
        st.write("Click 'Load Components' first")
    
    # Check portfolio
    if 'portfolio_manager' in st.session_state:
        portfolio = st.session_state.portfolio_manager
        summary = portfolio.get_portfolio_summary()
        st.write(f"**Available Cash:** ${summary['available_cash']:,.2f}")
        st.write(f"**Active Positions:** {summary['num_positions']}")
        st.write(f"**Total Portfolio Value:** ${summary['total_portfolio_value']:,.2f}")
    else:
        st.error("❌ Portfolio Manager: NOT INITIALIZED")
    
    # Force Trading Section
    st.subheader("🚀 Force Execute Trades")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Generate & Execute Signals", help="Force generate trading signals and execute"):
            if 'trading_engine' in st.session_state and 'news_df' in st.session_state:
                engine = st.session_state.trading_engine
                news_df = st.session_state.news_df
                
                # Force generate signals
                with st.spinner("Generating signals..."):
                    signals = engine.analyze_news_and_generate_signals(news_df)
                
                st.write(f"**Generated {len(signals)} trading signals**")
                
                if signals:
                    # Display signals
                    for i, signal in enumerate(signals[:3]):
                        st.write(f"Signal {i+1}: {signal['action']} {signal['symbol']} (Confidence: {signal['confidence']:.2f})")
                    
                    # Force execute
                    with st.spinner("Executing trades..."):
                        executed = engine.execute_trading_signals(signals[:3])
                    
                    st.write(f"**Executed {len(executed)} trades**")
                    for trade in executed:
                        st.success(f"✅ {trade['trade_result']['message']}")
                        
                    if len(executed) == 0:
                        st.warning("⚠️ No trades executed - check confidence thresholds")
                else:
                    st.warning("❌ No signals generated - check news sentiment scores")
            else:
                st.error("Trading engine or news data not available")
    
    with col2:
        # Manual trading
        st.write("**Manual Trade Execution:**")
        manual_symbol = st.selectbox("Select Stock", config.US_STOCKS, key="manual_stock")
        manual_confidence = st.slider("Confidence", 0.5, 1.0, 0.8, key="manual_conf")
        
        if st.button("💰 Execute Manual Buy", help="Force buy this stock"):
            if 'portfolio_manager' in st.session_state:
                portfolio = st.session_state.portfolio_manager
                try:
                    result = portfolio.buy_stock(
                        manual_symbol, 
                        manual_confidence, 
                        reason="Manual Force Buy"
                    )
                    if result['success']:
                        st.success(f"✅ {result['message']}")
                        st.experimental_rerun()  # Refresh to show new trade
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Manual trade error: {str(e)}")
            else:
                st.error("❌ Load components first!")

def main():
    st.set_page_config(
        page_title="AlphaTrading - US Market",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title(config.DASHBOARD_TITLE)
    st.markdown("**Advanced AI-Powered Trading with $5M Virtual Capital + Real NewsAPI Integration**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # NewsAPI Key Input
        st.subheader("📰 NewsAPI Integration")
        newsapi_key = st.text_input(
            "NewsAPI Key (optional)", 
            value=config.NEWSAPI_KEY,
            type="password",
            help="Get free key from newsapi.org for real news"
        )
        
        if newsapi_key:
            st.success("✅ NewsAPI key configured")
        else:
            st.info("ℹ️ Using synthetic news only")
        
        # Auto-refresh controls
        st.subheader("🔄 Data Refresh")
        auto_refresh_interval = st.selectbox(
            "Auto-refresh interval", 
            [1, 2, 3, 5, 10], 
            index=2,
            help="Minutes between automatic data refresh"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Force Refresh", help="Generate new data immediately"):
                st.cache_data.clear()
                force_refresh = True
            else:
                force_refresh = False
        
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear all cached data"):
                st.cache_data.clear()
                st.success("Cache cleared!")
    
    # System Initialization
    st.subheader("🔧 System Initialization")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Load Components", help="Initialize all trading system components"):
            with st.spinner("Loading trading system components..."):
                try:
                    components = load_components()
                    if components:
                        st.session_state.news_generator = components[0]
                        st.session_state.sentiment_analyzer = components[1] 
                        st.session_state.prediction_model = components[2]
                        st.session_state.portfolio_manager = components[3]
                        st.session_state.trading_engine = components[4]
                        st.success("✅ All components loaded successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to load components")
                except Exception as e:
                    st.error(f"❌ Error loading components: {str(e)}")

    with col2:
        # Show current status
        if 'trading_engine' in st.session_state:
            st.success("✅ Trading System: Ready")
            if st.button("▶️ Start Trading"):
                st.session_state.trading_engine.start_trading()
                st.success("Trading started!")
                st.experimental_rerun()
        else:
            st.warning("⚠️ Click 'Load Components' first")
    
    # Load enhanced news data
    with st.spinner("📰 Loading mixed news data (Real + Synthetic)..."):
        news_df = load_news_data_enhanced(
            force_refresh=force_refresh, 
            newsapi_key=newsapi_key
        )
        st.session_state.news_df = news_df  # Store for trading engine
    
    # Display data mix information
    if 'is_real' in news_df.columns:
        real_count = news_df['is_real'].sum() if 'is_real' in news_df.columns else 0
        synthetic_count = len(news_df) - real_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌐 Real News", f"{real_count}")
        with col2:
            st.metric("🤖 Synthetic News", f"{synthetic_count}")
        with col3:
            st.metric("📊 Total Articles", f"{len(news_df)}")
    
    # Add trading debug section
    debug_and_fix_trading()
    
    # Display recent news
    st.subheader("📰 Recent News Headlines")
    if not news_df.empty:
        for _, row in news_df.head(10).iterrows():
            real_flag = "🌐 REAL" if row.get('is_real', False) else "🤖 SYNTHETIC"
            st.write(f"{real_flag}: {row['headline']} ({row['symbol']})")

if __name__ == "__main__":
    main()
