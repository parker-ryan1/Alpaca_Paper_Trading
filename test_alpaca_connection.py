"""
Simple test to check if Alpaca connection works
"""
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get credentials
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = os.getenv('ALPACA_BASE_URL')

print("="*60)
print("TESTING ALPACA CONNECTION")
print("="*60)

print(f"\nAPI Key: {api_key[:10]}...{api_key[-5:]}")
print(f"Base URL: {base_url}")

try:
    # Initialize API
    api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    # Get account info
    account = api.get_account()
    
    print("\n" + "="*60)
    print("CONNECTION SUCCESSFUL!")
    print("="*60)
    
    print(f"\nYour Paper Trading Account:")
    print(f"   Status:          {account.status}")
    print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"   Cash:            ${float(account.cash):,.2f}")
    print(f"   Buying Power:    ${float(account.buying_power):,.2f}")
    
    # Check market status
    clock = api.get_clock()
    market_status = "OPEN" if clock.is_open else "CLOSED"
    print(f"\nMarket Status: {market_status}")
    
    if not clock.is_open:
        print(f"   Next Open:  {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")
    
    print("\n" + "="*60)
    print("YOU'RE ALL SET UP!")
    print("="*60)
    print("\nNext steps:")
    print("1. Wait for packages to finish installing")
    print("2. Run: python start_live_trading.py")
    print("3. Or: python main.py backtest --symbols AAPL --strategy macd")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n[ERROR] Connection failed: {e}")
    print("\nPlease check:")
    print("- API keys are correct in .env file")
    print("- You're using PAPER trading keys (not live!)")
    print("- Internet connection is working")
