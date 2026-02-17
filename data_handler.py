"""
Data Handler for fetching and managing market data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """Handles all data fetching and preprocessing"""
    
    def __init__(self):
        self.data_cache = {}
    
    def fetch_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean column names
                df.columns = df.columns.str.lower()
                
                # Add symbol column
                df['symbol'] = symbol
                
                data[symbol] = df
                self.data_cache[symbol] = df
                
                logger.info(f"Fetched {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return data
    
    def get_latest_data(
        self, 
        symbol: str, 
        period: str = '1mo'
    ) -> pd.DataFrame:
        """
        Get latest data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Period to fetch (1d, 5d, 1mo, 3mo, 1y, etc.)
        
        Returns:
            DataFrame with latest data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            df.columns = df.columns.str.lower()
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """
        Get current live price for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Current price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
        return None
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with fundamental data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for a DataFrame"""
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        return df
    
    def resample_data(
        self, 
        df: pd.DataFrame, 
        timeframe: str = '1W'
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe (1W, 1M, etc.)
        
        Returns:
            Resampled DataFrame
        """
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna()


if __name__ == "__main__":
    # Test the data handler
    handler = DataHandler()
    
    # Fetch data for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = handler.fetch_data(
        symbols=symbols,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    print(f"\nFetched data for {len(data)} symbols")
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows")
        print(df.head())
