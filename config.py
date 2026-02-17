"""
Configuration file for the trading system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Trading Parameters
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
COMMISSION = float(os.getenv('COMMISSION', 0.001))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))

# Alpaca Paper Trading
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Backtesting Parameters
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# Strategy Parameters
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_PERIOD = 20
BB_STD = 2

# Risk Management
MAX_POSITION_SIZE = 0.2  # Max 20% per position
STOP_LOSS_PCT = 0.05     # 5% stop loss
TAKE_PROFIT_PCT = 0.10   # 10% take profit
