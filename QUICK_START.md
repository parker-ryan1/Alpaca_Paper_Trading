# ⚡ QUICK START - Paper Trading in 5 Minutes

## 🎯 Two Ways to Start:

---

## 📍 METHOD 1: Instant Start (No Setup)

### Run this command:
```bash
python main.py paper --symbols AAPL --strategy macd --duration 60
```

**What happens:**
- Uses built-in simulator
- Fetches Yahoo Finance data
- Simulates trading for 60 minutes
- Shows results in console

**Done! You're paper trading!**

---

## 🚀 METHOD 2: Real Paper Trading with Alpaca (5 min setup)

### Step 1: Get Free API Keys (2 minutes)
1. Go to: **https://alpaca.markets/**
2. Click "Sign Up" → Choose "Paper Trading"
3. Verify email
4. Copy your API keys from dashboard

### Step 2: Add Keys to .env file (1 minute)
```bash
# Create .env file
copy .env.example .env

# Edit .env and add:
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

### Step 3: Run It! (30 seconds)
```bash
python start_live_trading.py
```

**Done! You're live paper trading with real broker API!**

---

## 🎮 Commands Cheat Sheet

### Backtest First (Recommended):
```bash
python main.py backtest --symbols AAPL --strategy macd --plot
```

### Built-in Paper Trading:
```bash
# Basic
python main.py paper --symbols AAPL --strategy macd

# Multiple stocks, 2 hours
python main.py paper --symbols AAPL MSFT GOOGL --strategy rsi --duration 120

# Custom settings
python main.py paper --symbols TSLA --strategy multi_factor --capital 50000 --interval 60
```

### Alpaca Paper Trading:
```bash
# Simple start
python start_live_trading.py

# Or use the class directly
python alpaca_trader.py
```

### List All Strategies:
```bash
python main.py list-strategies
```

---

## 📊 Available Strategies

- **macd** - MACD Crossover (momentum)
- **rsi** - RSI Mean Reversion (oversold/overbought)
- **bollinger** - Bollinger Bands (mean reversion)
- **ma_crossover** - Moving Average Cross (trend following)
- **multi_factor** - Multiple indicators combined

---

## 🛑 Stop Trading

Press **Ctrl+C** at any time to stop

---

## 📈 What You'll See

```
======================================================================
Iteration 5 - 2024-02-16 14:30:00
======================================================================
AAPL: No action (signal=0, position=none)
MSFT: 📈 BUY SIGNAL @ $415.20
Order FILLED: 48 MSFT @ $415.25

======================================================================
PAPER TRADING ACCOUNT SUMMARY
======================================================================
Cash:              $80,067.00
Positions Value:   $19,932.00
Total Value:       $99,999.00
Total Return:      -0.00%
Open Positions:    1
Total Trades:      2
======================================================================

Waiting 300 seconds...
```

---

## 🎯 Pro Tips

### 1. Always Backtest First
```bash
python main.py backtest --symbols AAPL --strategy macd --plot
```
See how the strategy performed historically before paper trading

### 2. Start with 1-2 Stocks
Don't overwhelm yourself

### 3. Use Longer Intervals
300 seconds (5 min) is better than 60 seconds

### 4. Check During Market Hours
Market is open 9:30 AM - 4:00 PM ET (Mon-Fri)

### 5. Monitor Performance
Watch for a few days, then adjust

---

## ⚠️ Troubleshooting

### "No module named 'alpaca_trade_api'"
```bash
pip install -r requirements.txt
```

### "Alpaca credentials not found"
Add your keys to `.env` file (see Method 2 above)

### "Market is closed"
Normal! System will wait for market to open

### Want to test immediately?
Use Method 1 (built-in simulator) - works 24/7

---

## 🎓 Learn More

- **Full Setup:** Read `SETUP_GUIDE.md`
- **All Features:** Read `README.md`
- **Strategy Details:** Look in `strategies.py`

---

## 📞 What's Next?

### After Paper Trading for a Week:

1. **Review results** - What worked? What didn't?
2. **Adjust parameters** - Try different strategies
3. **Optimize** - Fine-tune your approach
4. **Repeat** - Keep testing and improving

---

## 🎉 You're Ready!

### Choose your path:

**Quick Test:**
```bash
python main.py paper --symbols AAPL --strategy macd --duration 60
```

**Serious Testing:**
```bash
# Set up Alpaca (2 min), then:
python start_live_trading.py
```

**Backtest First:**
```bash
python main.py backtest --symbols AAPL --strategy macd --plot
```

---

**That's it! Start trading! 📈🚀**

*(Remember: This is paper trading - no real money involved!)*
