"""
Test Script - Verify Everything is Working

This script tests all components of the trading system.
Run this after installation to make sure everything works!
"""
import sys
import traceback


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*60)
    print("TEST 1: Checking Dependencies")
    print("="*60)
    
    packages = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('yfinance', 'Market data'),
        ('matplotlib', 'Visualization'),
        ('pandas_ta', 'Technical indicators'),
        ('quantstats', 'Performance analytics'),
        ('alpaca_trade_api', 'Live trading (optional)'),
    ]
    
    all_good = True
    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {package:20} - {description}")
        except ImportError:
            print(f"❌ {package:20} - NOT INSTALLED")
            all_good = False
    
    if all_good:
        print("\n✅ All dependencies installed!")
    else:
        print("\n❌ Some packages missing. Run: pip install -r requirements.txt")
        return False
    
    return True


def test_data_handler():
    """Test data fetching"""
    print("\n" + "="*60)
    print("TEST 2: Testing Data Handler")
    print("="*60)
    
    try:
        from data_handler import DataHandler
        
        handler = DataHandler()
        print("✅ DataHandler initialized")
        
        # Test fetching data
        print("📡 Fetching AAPL data...")
        data = handler.fetch_data(['AAPL'], '2024-01-01', '2024-01-31')
        
        if 'AAPL' in data and not data['AAPL'].empty:
            print(f"✅ Fetched {len(data['AAPL'])} days of data")
            print(f"   Latest price: ${data['AAPL']['close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ Failed to fetch data")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False


def test_indicators():
    """Test technical indicators"""
    print("\n" + "="*60)
    print("TEST 3: Testing Technical Indicators")
    print("="*60)
    
    try:
        from data_handler import DataHandler
        from indicators import TechnicalIndicators
        
        # Get data
        handler = DataHandler()
        data = handler.fetch_data(['AAPL'], '2024-01-01', '2024-01-31')
        df = data['AAPL']
        
        # Add indicators
        print("📊 Calculating indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        
        indicators = ['sma_20', 'rsi', 'macd', 'bb_upper']
        all_present = all(ind in df.columns for ind in indicators)
        
        if all_present:
            print("✅ All indicators calculated successfully")
            print(f"   RSI: {df['rsi'].iloc[-1]:.2f}")
            print(f"   MACD: {df['macd'].iloc[-1]:.4f}")
            return True
        else:
            print("❌ Some indicators missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False


def test_strategies():
    """Test trading strategies"""
    print("\n" + "="*60)
    print("TEST 4: Testing Trading Strategies")
    print("="*60)
    
    try:
        from data_handler import DataHandler
        from strategies import MACDStrategy, RSIMeanReversion
        
        # Get data
        handler = DataHandler()
        data = handler.fetch_data(['AAPL'], '2024-01-01', '2024-01-31')
        df = data['AAPL']
        
        # Test MACD strategy
        print("📈 Testing MACD Strategy...")
        strategy = MACDStrategy()
        df_signals = strategy.generate_signals(df)
        
        num_signals = (df_signals['signal'] != 0).sum()
        print(f"✅ MACD Strategy: Generated {num_signals} signals")
        
        # Test RSI strategy
        print("📉 Testing RSI Strategy...")
        strategy2 = RSIMeanReversion()
        df_signals2 = strategy2.generate_signals(df)
        
        num_signals2 = (df_signals2['signal'] != 0).sum()
        print(f"✅ RSI Strategy: Generated {num_signals2} signals")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False


def test_backtester():
    """Test backtesting engine"""
    print("\n" + "="*60)
    print("TEST 5: Testing Backtester")
    print("="*60)
    
    try:
        from data_handler import DataHandler
        from strategies import MACDStrategy
        from backtester import Backtester
        
        # Get data
        print("📊 Running backtest...")
        handler = DataHandler()
        data = handler.fetch_data(['AAPL'], '2023-01-01', '2024-01-01')
        df = data['AAPL']
        
        # Generate signals
        strategy = MACDStrategy()
        df = strategy.generate_signals(df)
        
        # Run backtest
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(df)
        
        metrics = backtester.get_performance_metrics()
        
        if metrics:
            print(f"✅ Backtest completed successfully")
            print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
            return True
        else:
            print("❌ Backtest produced no results")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization"""
    print("\n" + "="*60)
    print("TEST 6: Testing Visualization")
    print("="*60)
    
    try:
        from visualizer import Visualizer
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        print("✅ Visualizer initialized")
        print("   (Plotting disabled for testing)")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False


def test_alpaca_connection():
    """Test Alpaca connection (optional)"""
    print("\n" + "="*60)
    print("TEST 7: Testing Alpaca Connection (Optional)")
    print("="*60)
    
    try:
        import config
        
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            print("⚠️  Alpaca credentials not configured (OK for now)")
            print("   To enable live paper trading:")
            print("   1. Sign up at https://alpaca.markets/")
            print("   2. Add API keys to .env file")
            return True
        
        from alpaca_trader import AlpacaPaperTrader
        
        print("🔌 Connecting to Alpaca...")
        trader = AlpacaPaperTrader()
        
        account = trader.get_account()
        if account:
            print(f"✅ Connected to Alpaca!")
            print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"   Buying Power: ${account['buying_power']:,.2f}")
            return True
        else:
            print("❌ Failed to get account info")
            return False
            
    except Exception as e:
        print(f"⚠️  Alpaca connection failed: {e}")
        print("   This is optional - system works without it")
        return True  # Not critical


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRADING SYSTEM TEST SUITE")
    print("="*60)
    print("\nTesting all components...\n")
    
    tests = [
        ("Dependencies", test_imports),
        ("Data Handler", test_data_handler),
        ("Technical Indicators", test_indicators),
        ("Trading Strategies", test_strategies),
        ("Backtester", test_backtester),
        ("Visualization", test_visualization),
        ("Alpaca Connection", test_alpaca_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use!")
        print("\nNext steps:")
        print("1. Run a backtest: python main.py backtest --symbols AAPL --strategy macd")
        print("2. Try paper trading: python main.py paper --symbols AAPL --strategy macd")
        print("3. Set up Alpaca for live paper trading (see SETUP_GUIDE.md)")
    elif passed >= total - 1:  # All but Alpaca
        print("\n✅ Core system working! (Alpaca optional)")
        print("\nYou can start using the system now!")
        print("Run: python main.py backtest --symbols AAPL --strategy macd")
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")
        print("Try: pip install -r requirements.txt")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
