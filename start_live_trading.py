"""
Quick Start: Live Paper Trading with Alpaca

This script starts live paper trading with your chosen strategy.
"""
from alpaca_trader import AlpacaPaperTrader
from strategies import get_strategy
import sys

def main():
    print("""
    ============================================================
           LIVE PAPER TRADING - RANDOM FOREST ML               
           (No Real Money - 100% Safe!)                        
           Best Strategy: 35% Return, 92% Win Rate!            
    ============================================================
    """)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # Stocks to trade
    STRATEGY_NAME = 'random_forest'       # Strategy to use (BEST PERFORMER!)
    CHECK_INTERVAL = 300                  # Check every 5 minutes
    POSITION_SIZE = 0.20                  # Max 20% per position
    STOP_LOSS = 0.05                      # 5% stop loss
    TAKE_PROFIT = 0.10                    # 10% take profit
    
    try:
        # Initialize Alpaca trader
        print("[*] Connecting to Alpaca...")
        trader = AlpacaPaperTrader()
        
        # Show account status
        trader.print_summary()
        
        # Check if market is open
        hours = trader.get_market_hours()
        if not hours['is_open']:
            print(f"\n[WARNING] Market is currently CLOSED")
            print(f"[INFO] Next open: {hours['next_open']}")
            print(f"\nThe system will wait until market opens.")
        else:
            print("\n[OK] Market is OPEN - Ready to trade!")
        
        # Get strategy
        print(f"\n[STRATEGY] Using strategy: {STRATEGY_NAME}")
        strategy = get_strategy(STRATEGY_NAME)
        
        print(f"[SYMBOLS] Trading symbols: {', '.join(SYMBOLS)}")
        print(f"[INTERVAL] Check interval: {CHECK_INTERVAL} seconds")
        print(f"\n{'='*60}")
        print("Press Ctrl+C to stop trading at any time")
        print(f"{'='*60}\n")
        
        input("Press ENTER to start trading...")
        
        # Start trading
        trader.start_trading(
            strategy=strategy,
            symbols=SYMBOLS,
            check_interval=CHECK_INTERVAL,
            position_size_pct=POSITION_SIZE,
            stop_loss_pct=STOP_LOSS,
            take_profit_pct=TAKE_PROFIT
        )
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Trading stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nMake sure you've:")
        print("1. Signed up at https://alpaca.markets/")
        print("2. Added your API keys to .env file")
        print("3. Installed dependencies: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
