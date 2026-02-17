"""
Interactive Alpaca Setup Script
Helps you configure your Alpaca paper trading credentials
"""
import os
import sys
from pathlib import Path


def print_banner():
    print("""
    ===========================================================
    
            ALPACA PAPER TRADING SETUP
    
            Get your FREE $100,000 paper trading account!
    
    ===========================================================
    """)


def check_env_file():
    """Check if .env file exists"""
    env_path = Path('.env')
    if env_path.exists():
        print("[OK] .env file found")
        return True
    else:
        print("[!] .env file not found")
        return False


def show_instructions():
    """Show step-by-step instructions"""
    print("\n" + "="*60)
    print("STEP 1: Sign Up for Alpaca (2 minutes)")
    print("="*60)
    print("""
1. Go to: https://alpaca.markets/
2. Click the "Sign Up" button (top right)
3. Choose "Paper Trading Only" (it's FREE!)
4. Fill in your information
5. Verify your email address
6. You'll get $100,000 in paper money instantly!
    """)
    
    input("Press ENTER after you've signed up...")
    
    print("\n" + "="*60)
    print("STEP 2: Get Your API Keys")
    print("="*60)
    print("""
1. Log in to your Alpaca account
2. Go to the main dashboard
3. Look for "Your API Keys" in the left menu
4. Click "Generate New Keys" or "View" if keys exist
5. Make sure you're on the "Paper Trading" tab (NOT live!)
6. Copy both keys:
   - API Key ID (starts with PK...)
   - Secret Key (starts with...)
    """)
    
    input("Press ENTER when you have your keys ready...")


def get_api_keys():
    """Get API keys from user"""
    print("\n" + "="*60)
    print("STEP 3: Enter Your API Keys")
    print("="*60)
    
    print("\nIMPORTANT: Make sure these are PAPER TRADING keys!")
    print("(Paper keys usually start with 'PK')\n")
    
    api_key = input("Enter your API Key ID: ").strip()
    
    if not api_key or api_key == 'your_api_key_here':
        print("[ERROR] Invalid API key")
        return None, None
    
    secret_key = input("Enter your Secret Key: ").strip()
    
    if not secret_key or secret_key == 'your_secret_key_here':
        print("[ERROR] Invalid secret key")
        return None, None
    
    return api_key, secret_key


def save_credentials(api_key, secret_key):
    """Save credentials to .env file"""
    env_content = f"""# Alpaca Paper Trading Credentials
ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
INITIAL_CAPITAL=100000
COMMISSION=0.001
RISK_PER_TRADE=0.02
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("\n[OK] Credentials saved to .env file!")


def test_connection():
    """Test connection to Alpaca"""
    print("\n" + "="*60)
    print("STEP 4: Testing Connection")
    print("="*60)
    
    print("\nConnecting to Alpaca...")
    
    try:
        from alpaca_trader import AlpacaPaperTrader
        
        trader = AlpacaPaperTrader()
        account = trader.get_account()
        
        if account:
            print("\n" + "="*60)
            print("CONNECTION SUCCESSFUL!")
            print("="*60)
            print(f"\nYour Paper Trading Account:")
            print(f"   Status:          {account['status']}")
            print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"   Cash:            ${account['cash']:,.2f}")
            print(f"   Buying Power:    ${account['buying_power']:,.2f}")
            
            # Check market status
            hours = trader.get_market_hours()
            market_status = "OPEN" if hours['is_open'] else "CLOSED"
            print(f"\nMarket Status: {market_status}")
            
            if not hours['is_open']:
                print(f"   Next Open: {hours['next_open']}")
            
            print("\n" + "="*60)
            print("YOU'RE ALL SET UP!")
            print("="*60)
            print("\nNext steps:")
            print("1. Run a backtest first:")
            print("   python main.py backtest --symbols AAPL --strategy macd --plot")
            print("\n2. Start paper trading:")
            print("   python start_live_trading.py")
            print("\n3. Or customize your trading:")
            print("   python alpaca_trader.py")
            print("\n" + "="*60)
            
            return True
        else:
            print("\n[ERROR] Could not get account information")
            return False
            
    except ImportError:
        print("\n[ERROR] Required packages not installed")
        print("Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nPossible issues:")
        print("- Check if your API keys are correct")
        print("- Make sure you're using PAPER trading keys")
        print("- Verify your internet connection")
        return False


def main():
    """Main setup flow"""
    print_banner()
    
    print("This wizard will help you set up Alpaca paper trading.\n")
    
    # Show instructions
    show_instructions()
    
    # Get API keys
    api_key, secret_key = get_api_keys()
    
    if not api_key or not secret_key:
        print("\n❌ Setup cancelled. Please run this script again.")
        sys.exit(1)
    
    # Confirm
    print(f"\nAPI Key ID: {api_key[:10]}...{api_key[-5:]}")
    print(f"Secret Key: {secret_key[:10]}...{'*' * 10}")
    
    confirm = input("\nSave these credentials? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n[CANCELLED] Setup cancelled.")
        sys.exit(1)
    
    # Save credentials
    save_credentials(api_key, secret_key)
    
    # Test connection
    success = test_connection()
    
    if success:
        print("\nReady to start trading! Good luck!")
    else:
        print("\n[WARNING] Setup complete but connection test failed.")
        print("Check your credentials in the .env file and try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
