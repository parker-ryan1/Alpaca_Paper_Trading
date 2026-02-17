"""
Advanced Data Handler
Multiple data sources integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Data source imports (optional)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDataHandler:
    """
    Advanced Data Handler with Multiple Sources
    
    Supported data sources:
    - Yahoo Finance (yfinance)
    - Alpaca Markets
    - Polygon.io
    - Finnhub
    - IEX Cloud
    - Alpha Vantage
    """
    
    def __init__(self):
        self.alpaca_api_key = None
        self.alpaca_secret = None
        self.polygon_api_key = None
        self.finnhub_api_key = None
        self.iex_api_key = None
        self.alphavantage_api_key = None
    
    def set_alpaca_credentials(self, api_key: str, secret: str):
        """Set Alpaca API credentials"""
        self.alpaca_api_key = api_key
        self.alpaca_secret = secret
    
    def set_polygon_api_key(self, api_key: str):
        """Set Polygon.io API key"""
        self.polygon_api_key = api_key
    
    def set_finnhub_api_key(self, api_key: str):
        """Set Finnhub API key"""
        self.finnhub_api_key = api_key
    
    def fetch_from_yfinance(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        if not HAS_YFINANCE:
            logger.error("yfinance not installed")
            return {}
        
        data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol} from Yahoo Finance")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not df.empty:
                    # Standardize column names
                    df.columns = [col.lower() for col in df.columns]
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} rows for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return data
    
    def fetch_from_alpaca(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = '1Day'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Alpaca Markets"""
        if not HAS_ALPACA or not self.alpaca_api_key:
            logger.error("Alpaca not configured")
            return {}
        
        try:
            client = StockHistoricalDataClient(
                self.alpaca_api_key,
                self.alpaca_secret
            )
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day if timeframe == '1Day' else TimeFrame.Hour,
                start=datetime.strptime(start_date, '%Y-%m-%d'),
                end=datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            bars = client.get_stock_bars(request_params)
            
            data = {}
            for symbol in symbols:
                if symbol in bars.data:
                    df = bars.data[symbol].df
                    df.columns = [col.lower() for col in df.columns]
                    data[symbol] = df
            
            return data
        
        except Exception as e:
            logger.error(f"Alpaca error: {e}")
            return {}
    
    def fetch_from_polygon(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Polygon.io"""
        if not HAS_REQUESTS or not self.polygon_api_key:
            logger.error("Polygon.io not configured")
            return {}
        
        data = {}
        
        for symbol in symbols:
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
                params = {'apiKey': self.polygon_api_key}
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    json_data = response.json()
                    
                    if 'results' in json_data:
                        df = pd.DataFrame(json_data['results'])
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                        df = df.set_index('timestamp')
                        
                        # Rename columns to standard format
                        df = df.rename(columns={
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume'
                        })
                        
                        data[symbol] = df[['open', 'high', 'low', 'close', 'volume']]
            
            except Exception as e:
                logger.error(f"Polygon error for {symbol}: {e}")
        
        return data
    
    def fetch_from_finnhub(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from Finnhub"""
        if not HAS_REQUESTS or not self.finnhub_api_key:
            logger.error("Finnhub not configured")
            return {}
        
        data = {}
        
        for symbol in symbols:
            try:
                # Convert dates to timestamps
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
                
                url = "https://finnhub.io/api/v1/stock/candle"
                params = {
                    'symbol': symbol,
                    'resolution': 'D',
                    'from': start_ts,
                    'to': end_ts,
                    'token': self.finnhub_api_key
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    json_data = response.json()
                    
                    if json_data.get('s') == 'ok':
                        df = pd.DataFrame({
                            'timestamp': pd.to_datetime(json_data['t'], unit='s'),
                            'open': json_data['o'],
                            'high': json_data['h'],
                            'low': json_data['l'],
                            'close': json_data['c'],
                            'volume': json_data['v']
                        })
                        
                        df = df.set_index('timestamp')
                        data[symbol] = df
            
            except Exception as e:
                logger.error(f"Finnhub error for {symbol}: {e}")
        
        return data
    
    def fetch_fundamental_data(self, symbol: str) -> Dict:
        """
        Fetch fundamental data (financials, ratios, etc.)
        
        Uses multiple sources for comprehensive data
        """
        if not HAS_YFINANCE:
            return {}
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return fundamentals
        
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def fetch_options_data(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain data"""
        if not HAS_YFINANCE:
            return pd.DataFrame()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return pd.DataFrame()
            
            # Get options for nearest expiration
            options = ticker.option_chain(expirations[0])
            
            # Combine calls and puts
            calls = options.calls
            calls['type'] = 'call'
            puts = options.puts
            puts['type'] = 'put'
            
            df = pd.concat([calls, puts], ignore_index=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_calendar(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """Fetch earnings calendar"""
        if not HAS_YFINANCE:
            return pd.DataFrame()
        
        calendar_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                earnings = ticker.calendar
                
                if earnings is not None and len(earnings) > 0:
                    earnings_date = earnings.get('Earnings Date', [None])[0]
                    
                    if earnings_date:
                        calendar_data.append({
                            'symbol': symbol,
                            'earnings_date': earnings_date,
                            'eps_estimate': earnings.get('EPS Estimate', None)
                        })
            
            except Exception as e:
                logger.error(f"Error fetching earnings for {symbol}: {e}")
        
        return pd.DataFrame(calendar_data)
    
    def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P500, Dow, Nasdaq, VIX
        names = ['S&P 500', 'Dow Jones', 'Nasdaq', 'VIX']
        
        overview = {}
        
        for symbol, name in zip(indices, names):
            try:
                data = self.fetch_from_yfinance(
                    [symbol],
                    (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d')
                )
                
                if symbol in data:
                    df = data[symbol]
                    current = df['close'].iloc[-1]
                    previous = df['close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    overview[name] = {
                        'current': current,
                        'change': change
                    }
            
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return overview
    
    def fetch_with_fallback(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        primary_source: str = 'yfinance'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data with automatic fallback to alternative sources
        
        Priority: yfinance -> alpaca -> polygon -> finnhub
        """
        sources = {
            'yfinance': self.fetch_from_yfinance,
            'alpaca': self.fetch_from_alpaca,
            'polygon': self.fetch_from_polygon,
            'finnhub': self.fetch_from_finnhub
        }
        
        # Try primary source first
        logger.info(f"Fetching from primary source: {primary_source}")
        data = sources[primary_source](symbols, start_date, end_date)
        
        # Check for missing symbols
        missing = [s for s in symbols if s not in data]
        
        # Try fallback sources
        if missing:
            logger.info(f"Attempting fallback for {len(missing)} symbols")
            
            for source_name, source_func in sources.items():
                if source_name == primary_source:
                    continue
                
                fallback_data = source_func(missing, start_date, end_date)
                data.update(fallback_data)
                
                missing = [s for s in missing if s not in fallback_data]
                
                if not missing:
                    break
        
        return data


if __name__ == "__main__":
    # Test advanced data handler
    handler = AdvancedDataHandler()
    
    # Test Yahoo Finance
    if HAS_YFINANCE:
        data = handler.fetch_from_yfinance(
            ['AAPL', 'MSFT'],
            '2024-01-01',
            '2024-02-01'
        )
        
        print("Yahoo Finance Data:")
        for symbol, df in data.items():
            print(f"\n{symbol}: {len(df)} rows")
            print(df.head())
        
        # Test fundamentals
        fundamentals = handler.fetch_fundamental_data('AAPL')
        print("\nFundamentals for AAPL:")
        for key, value in fundamentals.items():
            print(f"  {key}: {value}")
        
        # Test market overview
        overview = handler.get_market_overview()
        print("\nMarket Overview:")
        for name, data in overview.items():
            print(f"  {name}: {data['current']:.2f} ({data['change']:+.2f}%)")
