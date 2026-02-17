# 🤖 Algorithmic Trading System with ML & Stochastic Processes

Professional-grade algorithmic paper trading system featuring **13 trading strategies** including advanced Machine Learning and Stochastic Process models. Built with Python and integrated with Alpaca Paper Trading API.

## 🚀 Features

- **13 Professional Trading Strategies**
  - 5 Classic Technical Analysis strategies (MACD, RSI, Bollinger Bands, etc.)
  - 3 Machine Learning strategies (Random Forest, Gradient Boosting, Ensemble)
  - 5 Stochastic Process strategies (Ornstein-Uhlenbeck, Kalman Filter, etc.)

- **Comprehensive Backtesting Engine**
  - Historical performance analysis
  - Risk metrics (Sharpe, Sortino, Max Drawdown)
  - Trade-by-trade logging
  - Visual performance charts

- **Live Paper Trading**
  - Integration with Alpaca Paper Trading API
  - Automated signal generation
  - Position management with stop-loss/take-profit
  - Real-time monitoring

- **Advanced Analytics**
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
  - Performance metrics and reporting
  - Risk management tools

## 🏆 Best Performing Strategy

**Random Forest ML** achieved exceptional results in backtests (AAPL 2022-2024):
- **Total Return:** 35.19%
- **Win Rate:** 92.19%
- **Sharpe Ratio:** 4.87
- **Max Drawdown:** -0.30%

## 📋 Requirements

```bash
Python 3.12+
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/parker-ryan1/Alpaca_Paper_Trading.git
cd Alpaca_Paper_Trading
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Alpaca API credentials:**
```bash
cp .env.example .env
# Edit .env and add your Alpaca API keys
```

## 🎯 Quick Start

### List Available Strategies
```bash
python main.py list-strategies
```

### Run a Backtest
```bash
# Classic strategy
python main.py backtest --symbols AAPL --strategy macd

# ML strategy (best performer)
python main.py backtest --symbols AAPL --strategy random_forest

# Stochastic strategy
python main.py backtest --symbols AAPL --strategy kalman_filter
```

### Start Paper Trading
```bash
# Automatic start (no input required)
python start_rf_trading.py

# Or with prompts
python start_live_trading.py
```

## 📊 Available Strategies

### Classic Technical Analysis (5)
1. **MACD Crossover** (`macd`) - Trend following
2. **RSI Mean Reversion** (`rsi`) - Oversold/overbought
3. **Bollinger Bands** (`bollinger`) - Volatility breakouts
4. **MA Crossover** (`ma_crossover`) - Moving average trends
5. **Multi-Factor** (`multi_factor`) - Combined signals

### Machine Learning (3)
1. **Random Forest** (`random_forest`) ⭐ - Ensemble of decision trees
2. **Gradient Boosting** (`gradient_boosting`) - Sequential learning
3. **Ensemble ML** (`ensemble_ml`) - Combined ML models

### Stochastic Processes (5)
1. **Ornstein-Uhlenbeck** (`ornstein_uhlenbeck`) - Mean reversion SDE
2. **Kalman Filter** (`kalman_filter`) - State-space estimation
3. **Volatility Clustering** (`volatility_clustering`) - GARCH-inspired
4. **Momentum-Reversal** (`momentum_reversal`) - Dual timeframe
5. **Jump Diffusion** (`jump_diffusion`) - Event-based trading

## 📈 Performance Comparison (AAPL 2022-2024)

| Strategy | Return | Sharpe | Win Rate | Trades |
|----------|--------|--------|----------|--------|
| Random Forest ⭐ | 35.19% | 4.87 | 92.19% | 128 |
| MACD | 9.44% | 1.63 | 70.59% | 17 |
| Kalman Filter | 3.32% | -0.23 | 66.80% | 506 |
| O-U Process | 0.53% | -0.44 | 50.00% | 16 |

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[Setup Guide](SETUP_GUIDE.md)** - Detailed Alpaca integration
- **[All Strategies](ALL_STRATEGIES.md)** - Complete strategy reference
- **[Exotic Strategies](EXOTIC_STRATEGIES_GUIDE.md)** - ML & stochastic models
- **[Performance Comparison](STRATEGY_COMPARISON.md)** - Strategy analysis

## 🏗️ Project Structure

```
quantfinnance/
├── main.py                      # Main CLI interface
├── config.py                    # Configuration settings
├── data_handler.py              # Market data fetching
├── indicators.py                # Technical indicators
├── strategies.py                # Classic strategies
├── ml_strategies.py             # Machine Learning strategies
├── stochastic_strategies.py     # Stochastic process strategies
├── backtester.py                # Backtesting engine
├── paper_trader.py              # Simulated trading
├── alpaca_trader.py             # Alpaca API integration
├── visualizer.py                # Performance visualization
├── start_rf_trading.py          # Auto-start ML trading
├── start_live_trading.py        # Interactive trading start
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment template
```

## 🔐 Security

- Never commit your `.env` file with API keys
- Use paper trading accounts for testing
- Always test strategies thoroughly before live trading
- Keep your API keys secure

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Always conduct thorough testing before live trading
- Use paper trading to validate strategies
- Understand the risks involved in algorithmic trading
- Not financial advice - trade at your own risk

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [Alpaca Trading API](https://alpaca.markets/)
- [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)
- [scikit-learn](https://scikit-learn.org/)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for quantitative traders and ML enthusiasts**

**⭐ If you find this useful, please star the repository!**
