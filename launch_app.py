"""
Launch Script for Algorithmic Trading Dashboard
Starts the Streamlit web application
"""
import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n📦 Installing required packages...")
        print("Run: pip install streamlit plotly textblob PyPortfolioOpt")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     ALGORITHMIC TRADING DASHBOARD - WEB APPLICATION        ║
    ║                                                            ║
    ║     🤖 13 Trading Strategies                               ║
    ║     📊 Live Backtesting                                    ║
    ║     💼 Portfolio Management                                ║
    ║     🔬 Risk Analytics                                      ║
    ║     📰 Sentiment Analysis                                  ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    if not check_dependencies():
        print("\n⚠️  Please install required packages first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🚀 Launching Streamlit dashboard...")
    print("📱 Dashboard will open in your browser")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n⏹️  Dashboard stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
