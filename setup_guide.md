# 🚀 Live Paper Trading Setup Guide

## What is Paper Trading?

**Paper trading** = Simulated live trading with **REAL market data** but **NO real money**. Perfect for:
- Testing your strategies safely
- Learning algorithmic trading
- Building confidence before live trading

---

## 📋 Two Options for Paper Trading

### **Option 1: Built-in Simulator** (Simple, no signup needed)
### **Option 2: Alpaca Integration** (Real broker API, more realistic)

---

## 🎯 Option 1: Built-in Paper Trading (Easy Start)

### Run immediately:
```bash
python main.py paper --symbols AAPL MSFT --strategy macd --duration 120
```

### What it does:
- ✅ Uses Yahoo Finance data
- ✅ Simulates order execution
- ✅ Tracks your portfolio
- ✅ No signup required
- ⚠️  Simplified execution model

---

## 🚀 Option 2: Alpaca Paper Trading (Recommended)

### Step 1: Sign Up for Alpaca (FREE)

1. Go to **https://alpaca.markets/**
2. Click **"Sign Up"**
3. Choose **"Paper Trading Only"** (free, no credit card needed)
4. Verify your email
5. You'll get $100,000 in paper money!

### Step 2: Get Your API Keys

1. Log in to Alpaca dashboard
2. Go to **"Your API Keys"** in the menu
3. Generate **Paper Trading** keys (NOT live trading!)
4. Copy:
   - API Key ID
   - Secret Key

### Step 3: Configure Your Environment

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your keys:
```env
# Alpaca Paper Trading Credentials
ALPACA_API_KEY=PKA...your_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Test Connection

```bash
python alpaca_trader.py
```

You should see your account info!

### Step 6: Start Live Paper Trading! 🎉

```bash
python start_live_trading.py
```

Or customize it:
```python
from alpaca_trader import AlpacaPaperTrader
from strategies import MACDStrategy

# Initialize
trader = AlpacaPaperTrader()

# Check account
trader.print_summary()

# Start trading
strategy = MACDStrategy()
trader.start_trading(
    strategy=strategy,
    symbols=['AAPL', 'MSFT', 'TSLA'],
    check_interval=300,  # Check every 5 minutes
    position_size_pct=0.15,  # Max 15% per position
    stop_loss_pct=0.05,      # 5% stop loss
    take_profit_pct=0.10     # 10% take profit
)
```

---

## 📊 What Happens During Paper Trading?

### Every Check Interval (e.g., 5 minutes):

1. **Fetch Latest Data** - Gets current market prices
2. **Calculate Indicators** - RSI, MACD, Bollinger Bands, etc.
3. **Generate Signals** - Strategy decides to BUY/SELL/HOLD
4. **Execute Orders** - Places orders with Alpaca (or simulates)
5. **Monitor Positions** - Checks stop-loss and take-profit
6. **Report Status** - Shows your portfolio value and P&L

### Example Console Output:

```
======================================================================
Iteration 1 - 2024-02-16 14:30:00
======================================================================
Fetching data for AAPL
📈 BUY SIGNAL for AAPL @ $182.50
BUY order submitted: 100 shares of AAPL (Order ID: 123)
Order FILLED: 100 AAPL @ $182.52
Stop loss placed at $173.39
Take profit placed at $200.77

======================================================================
ALPACA PAPER TRADING ACCOUNT
======================================================================
Status:           ACTIVE
Portfolio Value:  $99,824.00
Cash:             $81,572.00
Buying Power:     $81,572.00

======================================================================
OPEN POSITIONS
======================================================================

AAPL: 100 shares
  Entry Price:  $182.52
  Current:      $182.50
  Market Value: $18,250.00
  P&L:          -$2.00 (-0.01%)
======================================================================

Waiting 300 seconds...
```

---

## 🛡️ Risk Management Features

### Automatic Stop-Loss
If price drops 5% below entry, auto-sells to limit losses

### Take-Profit
If price rises 10% above entry, auto-sells to lock in profits

### Position Sizing
Never risks more than 20% of portfolio on one position

### Market Hours
Only trades when market is open (9:30 AM - 4:00 PM ET)

---

## 🎮 Control Your Paper Trading

### Start Trading:
```bash
python start_live_trading.py
```

### Stop Trading:
Press `Ctrl+C` at any time

### Close All Positions:
```python
trader.close_all_positions()
```

### Cancel All Orders:
```python
trader.cancel_all_orders()
```

### Check Status Anytime:
```python
trader.print_summary()
```

---

## 📈 Monitor Your Performance

### In Real-Time:
The console shows updates every check interval

### Alpaca Dashboard:
- Go to https://app.alpaca.markets/
- View your paper trading account
- See all orders, positions, and history
- Beautiful charts and analytics

### Export Results:
All trades are logged. You can analyze performance later.

---

## 🔧 Customization

### Change Strategy:
Edit `start_live_trading.py`:
```python
STRATEGY_NAME = 'rsi'  # or 'bollinger', 'ma_crossover', 'multi_factor'
```

### Change Symbols:
```python
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
```

### Change Check Interval:
```python
CHECK_INTERVAL = 60  # Check every 1 minute (be careful with rate limits!)
```

### Adjust Risk:
```python
POSITION_SIZE = 0.10  # Max 10% per position
STOP_LOSS = 0.03      # 3% stop loss
TAKE_PROFIT = 0.15    # 15% take profit
```

---

## ⚡ Pro Tips

### 1. Start Small
Begin with 2-3 stocks, then expand

### 2. Use Longer Intervals
5-10 minute intervals are better than 1-minute (less noise)

### 3. Test Multiple Strategies
Run different strategies and compare results

### 4. Monitor Market Hours
Market is closed nights, weekends, and holidays

### 5. Check Performance Weekly
Review what worked and what didn't

### 6. Backtest First
Always backtest before paper trading:
```bash
python main.py backtest --symbols AAPL --strategy macd --plot
```

---

## 🚨 Common Issues

### "Error: Alpaca credentials not found"
**Solution:** Add your API keys to `.env` file

### "Market is closed"
**Solution:** Normal! Market opens at 9:30 AM ET. The system will wait.

### "Order rejected"
**Solution:** Check if you have enough buying power

### "Connection timeout"
**Solution:** Check your internet connection

---

## 📊 Compare: Built-in vs Alpaca

| Feature | Built-in | Alpaca |
|---------|----------|--------|
| Setup | None | 5 minutes |
| Realism | Medium | High |
| Order Types | Market only | Market, Limit, Stop, etc. |
| Dashboard | No | Yes |
| Data Quality | Good | Excellent |
| Cost | Free | Free |
| Best For | Quick tests | Serious testing |

---

## 🎯 Next Steps

1. ✅ Set up Alpaca account
2. ✅ Configure `.env` file
3. ✅ Test connection
4. ✅ Run backtest first
5. ✅ Start paper trading
6. ✅ Monitor for a week
7. ✅ Analyze results
8. ✅ Optimize strategy
9. ✅ Repeat!

---

## 📚 Learning Resources

### Understanding Strategies:
- Read `strategies.py` comments
- Check `README.md` for strategy details

### Alpaca Documentation:
- https://alpaca.markets/docs/

### Trading Hours:
- Market: 9:30 AM - 4:00 PM ET (Mon-Fri)
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

---

## ⚠️ Important Reminders

- **This is PAPER TRADING** - No real money involved
- **Past performance ≠ future results**
- **Always test thoroughly** before considering live trading
- **Markets are risky** - Never invest more than you can afford to lose
- **This is for education** - Not financial advice

---

## 🎉 You're Ready!

Start your paper trading journey:

```bash
python start_live_trading.py
```

**Happy Trading! 📈🚀**

---

## 💬 Need Help?

1. Check the `README.md`
2. Review example code
3. Test with backtesting first
4. Start with simple strategies

Remember: The goal is to **learn and improve**, not to make quick profits!
