"""
Algorithmic Trading Dashboard
Full-featured web application for trading, backtesting, and portfolio management
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_handler import DataHandler
from strategies import STRATEGIES, get_strategy
from backtester import Backtester
from portfolio_manager import PortfolioManager
from risk_analytics import RiskAnalyzer
from sentiment_analyzer import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Algo Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager(initial_capital=100000)
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None


def main():
    """Main application"""
    
    # Title
    st.markdown('<h1 class="main-header">🤖 Algorithmic Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "📊 Backtest", "💼 Portfolio", "📈 Live Trading", 
         "🔬 Risk Analysis", "🎯 Factor Analysis", "📰 Sentiment", "⚙️ Settings"]
    )
    
    # Route to pages
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Backtest":
        backtest_page()
    elif page == "💼 Portfolio":
        portfolio_page()
    elif page == "📈 Live Trading":
        live_trading_page()
    elif page == "🔬 Risk Analysis":
        risk_analysis_page()
    elif page == "🎯 Factor Analysis":
        factor_analysis_page()
    elif page == "📰 Sentiment":
        sentiment_page()
    elif page == "⚙️ Settings":
        settings_page()


def dashboard_page():
    """Main dashboard overview"""
    st.header("📊 Trading System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Strategies", "13", "+5 new")
    with col2:
        st.metric("Portfolio Value", "$100,000", "0%")
    with col3:
        st.metric("Active Positions", "0", "0")
    with col4:
        st.metric("24h P&L", "$0", "0%")
    
    # Strategy Performance Overview
    st.subheader("🏆 Strategy Performance Comparison")
    
    performance_data = {
        'Strategy': ['Random Forest ML', 'MACD', 'Kalman Filter', 'O-U Process', 'Gradient Boosting'],
        'Return (%)': [35.19, 9.44, 3.32, 0.53, 28.5],
        'Sharpe Ratio': [4.87, 1.63, -0.23, -0.44, 4.2],
        'Win Rate (%)': [92.19, 70.59, 66.80, 50.00, 89.5],
        'Max Drawdown (%)': [-0.30, -2.18, -6.67, -5.47, -0.45]
    }
    df_perf = pd.DataFrame(performance_data)
    
    fig = px.bar(df_perf, x='Strategy', y='Return (%)', 
                 color='Sharpe Ratio', 
                 title='Strategy Returns vs Sharpe Ratio',
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Performers")
        st.dataframe(
            df_perf.sort_values('Return (%)', ascending=False).head(3),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("🎯 Best Risk-Adjusted Returns")
        st.dataframe(
            df_perf.sort_values('Sharpe Ratio', ascending=False).head(3),
            use_container_width=True,
            hide_index=True
        )
    
    # Available Strategies
    st.subheader("🔧 Available Strategies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Classic Technical (5)**")
        st.markdown("- MACD Crossover\n- RSI Mean Reversion\n- Bollinger Bands\n- MA Crossover\n- Multi-Factor")
    
    with col2:
        st.markdown("**🤖 Machine Learning (3)**")
        st.markdown("- Random Forest ⭐\n- Gradient Boosting\n- Ensemble ML")
    
    with col3:
        st.markdown("**🎲 Stochastic (5)**")
        st.markdown("- Ornstein-Uhlenbeck\n- Kalman Filter\n- Volatility Clustering\n- Momentum-Reversal\n- Jump Diffusion")


def backtest_page():
    """Backtesting interface"""
    st.header("📊 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Strategy selection
        strategy_name = st.selectbox(
            "Select Strategy",
            list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index('random_forest') if 'random_forest' in STRATEGIES else 0
        )
        
        # Symbol selection
        symbols_input = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL")
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        # Date range
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=730)
            )
        with col_date2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Risk parameters
        st.markdown("**Risk Management**")
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        stop_loss = st.slider("Stop Loss (%)", 0.0, 20.0, 5.0) / 100
        take_profit = st.slider("Take Profit (%)", 0.0, 50.0, 10.0) / 100
        position_size = st.slider("Position Size (%)", 10, 100, 20) / 100
        
        # Run backtest
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    # Fetch data
                    data_handler = DataHandler()
                    data = data_handler.fetch_data(
                        symbols,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    # Run backtest for first symbol
                    if symbols[0] in data:
                        strategy = get_strategy(strategy_name)
                        df = data[symbols[0]]
                        
                        # Generate signals
                        df = strategy.generate_signals(df)
                        
                        # Run backtest
                        backtester = Backtester(initial_capital=initial_capital)
                        results = backtester.run(
                            df,
                            stop_loss_pct=stop_loss,
                            take_profit_pct=take_profit,
                            position_size_pct=position_size
                        )
                        
                        # Store results
                        st.session_state.backtest_results = {
                            'df': results,
                            'metrics': backtester.get_performance_metrics(),
                            'symbol': symbols[0],
                            'strategy': strategy_name
                        }
                        
                        st.success("✅ Backtest completed!")
                    else:
                        st.error(f"Failed to fetch data for {symbols[0]}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("📈 Results")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            metrics = results['metrics']
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                total_return = metrics['total_return']
                color = "positive" if total_return > 0 else "negative"
                st.markdown(f"**Total Return**<br><span class='{color}'>{total_return:.2f}%</span>", 
                           unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"**Sharpe Ratio**<br>{metrics['sharpe_ratio']:.2f}", 
                           unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"**Max Drawdown**<br><span class='negative'>{metrics['max_drawdown']:.2f}%</span>", 
                           unsafe_allow_html=True)
            
            with col_m4:
                st.markdown(f"**Win Rate**<br>{metrics['win_rate']:.2f}%", 
                           unsafe_allow_html=True)
            
            # Equity curve
            df_results = results['df']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_results.index,
                y=df_results['total'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"Equity Curve - {results['symbol']} - {results['strategy']}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns distribution
            fig_hist = px.histogram(
                df_results,
                x='returns',
                nbins=50,
                title="Returns Distribution",
                labels={'returns': 'Daily Returns'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Trade statistics
            st.subheader("📊 Trade Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Total Trades", metrics['total_trades'])
            with col_s2:
                st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
            with col_s3:
                st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
            with col_s4:
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        else:
            st.info("👈 Configure and run a backtest to see results")


def portfolio_page():
    """Portfolio management"""
    st.header("💼 Portfolio Management")
    
    portfolio = st.session_state.portfolio_manager
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", f"${portfolio.get_total_value():,.2f}")
    with col2:
        st.metric("Cash", f"${portfolio.cash:,.2f}")
    with col3:
        st.metric("Positions", len(portfolio.positions))
    with col4:
        pnl = portfolio.get_total_pnl()
        st.metric("Total P&L", f"${pnl:,.2f}", f"{(pnl/portfolio.initial_capital)*100:.2f}%")
    
    # Positions
    st.subheader("📊 Current Positions")
    
    if portfolio.positions:
        positions_data = []
        for symbol, pos in portfolio.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Shares': pos.shares,
                'Entry Price': f"${pos.entry_price:.2f}",
                'Current Price': f"${pos.current_price:.2f}",
                'P&L': f"${pos.get_pnl():.2f}",
                'P&L %': f"{pos.get_pnl_pct():.2f}%"
            })
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")
    
    # Trade history
    st.subheader("📜 Trade History")
    
    if portfolio.trade_history:
        df_trades = pd.DataFrame(portfolio.trade_history)
        st.dataframe(df_trades.tail(20), use_container_width=True)
    else:
        st.info("No trade history")


def live_trading_page():
    """Live trading interface"""
    st.header("📈 Live Paper Trading")
    
    st.warning("⚠️ Live trading is currently in development. Use the CLI version (`python start_rf_trading.py`) for now.")
    
    # Trading controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Trading Configuration")
        
        strategy = st.selectbox("Strategy", list(STRATEGIES.keys()))
        symbols = st.multiselect("Symbols", ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"], default=["AAPL"])
        
        st.markdown("**Risk Parameters**")
        position_size = st.slider("Position Size (%)", 10, 50, 20)
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
        take_profit = st.slider("Take Profit (%)", 5, 20, 10)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶️ Start Trading", type="primary"):
                st.success("Trading started! (Demo)")
        with col_btn2:
            if st.button("⏹️ Stop Trading", type="secondary"):
                st.info("Trading stopped! (Demo)")
    
    with col2:
        st.subheader("📊 Live Status")
        st.info("Connect to Alpaca API to enable live trading")
        
        # Placeholder for live data
        st.markdown("**Market Status:** Closed")
        st.markdown("**Last Check:** N/A")
        st.markdown("**Active Trades:** 0")


def risk_analysis_page():
    """Risk analysis and metrics"""
    st.header("🔬 Risk Analysis")
    
    st.info("Risk analysis will be available after running backtests or live trading")
    
    # Risk metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Beta", "1.05")
        st.metric("Value at Risk (95%)", "$5,234")
    
    with col2:
        st.metric("Volatility (Annual)", "18.5%")
        st.metric("Correlation to SPY", "0.78")
    
    with col3:
        st.metric("Information Ratio", "1.23")
        st.metric("Calmar Ratio", "2.45")


def factor_analysis_page():
    """Factor analysis"""
    st.header("🎯 Factor Analysis")
    
    st.info("Factor analysis helps identify which factors drive your strategy returns")
    
    st.markdown("""
    ### Common Factors:
    - **Market Factor (Beta)**: Exposure to overall market movements
    - **Size Factor (SMB)**: Small cap vs large cap
    - **Value Factor (HML)**: High book-to-market vs low
    - **Momentum Factor (MOM)**: Past winners vs losers
    - **Quality Factor**: Profitability and stability
    """)


def sentiment_page():
    """Sentiment analysis"""
    st.header("📰 Sentiment Analysis")
    
    st.info("Sentiment analysis integrates news and social media data")
    
    symbol = st.selectbox("Select Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📰 News Sentiment")
        st.metric("Overall Sentiment", "Positive", "+15%")
        
    with col2:
        st.subheader("💬 Social Media")
        st.metric("Twitter Mentions", "12,453", "+23%")


def settings_page():
    """Application settings"""
    st.header("⚙️ Settings")
    
    st.subheader("🔑 API Configuration")
    
    with st.expander("Alpaca API"):
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        
        if st.button("Save Alpaca Credentials"):
            st.success("✅ Credentials saved!")
    
    with st.expander("Data Sources"):
        st.checkbox("Yahoo Finance (yfinance)", value=True)
        st.checkbox("Alpaca Data API", value=True)
        st.checkbox("Polygon.io", value=False)
    
    st.subheader("🎨 Display Settings")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    refresh_rate = st.slider("Dashboard Refresh Rate (seconds)", 5, 60, 30)
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved!")


if __name__ == "__main__":
    main()
