"""
Technical Indicators using pandas_ta
"""
import pandas as pd
import numpy as np
from typing import Tuple

# Make pandas_ta optional
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not installed. Some indicators may not work.")
    print("Install with: pip install pandas-ta")


class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all common technical indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Moving Averages
        df = TechnicalIndicators.add_moving_averages(df)
        
        # MACD
        df = TechnicalIndicators.add_macd(df)
        
        # RSI
        df = TechnicalIndicators.add_rsi(df)
        
        # Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df)
        
        # Stochastic Oscillator
        df = TechnicalIndicators.add_stochastic(df)
        
        # ATR (Average True Range)
        df = TechnicalIndicators.add_atr(df)
        
        # Volume indicators
        df = TechnicalIndicators.add_volume_indicators(df)
        
        return df
    
    @staticmethod
    def add_moving_averages(
        df: pd.DataFrame,
        periods: list = [10, 20, 50, 200]
    ) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages"""
        df = df.copy()
        
        for period in periods:
            if not HAS_PANDAS_TA:
                # Manual calculation
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            else:
                sma_result = ta.sma(df['close'], length=period)
                ema_result = ta.ema(df['close'], length=period)
                
                df[f'sma_{period}'] = sma_result if isinstance(sma_result, pd.Series) else sma_result.iloc[:, 0]
                df[f'ema_{period}'] = ema_result if isinstance(ema_result, pd.Series) else ema_result.iloc[:, 0]
        
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicator"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual MACD calculation
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        else:
            macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
            
            if macd is not None:
                cols = macd.columns.tolist()
                
                # Find correct column names
                macd_col = [c for c in cols if 'MACD_' in c and 'h' not in c and 's' not in c.lower()]
                signal_col = [c for c in cols if 'MACD' in c and ('s' in c.lower() or 'signal' in c.lower())]
                hist_col = [c for c in cols if 'MACD' in c and 'h' in c]
                
                if macd_col and signal_col and hist_col:
                    df['macd'] = macd[macd_col[0]]
                    df['macd_signal'] = macd[signal_col[0]]
                    df['macd_hist'] = macd[hist_col[0]]
                else:
                    # Fallback
                    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                    df['macd'] = ema_fast - ema_slow
                    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            rsi_result = ta.rsi(df['close'], length=period)
            if isinstance(rsi_result, pd.DataFrame):
                cols = [c for c in rsi_result.columns if 'RSI' in c]
                df['rsi'] = rsi_result[cols[0]] if cols else rsi_result.iloc[:, 0]
            else:
                df['rsi'] = rsi_result
        
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2
    ) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual calculation
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            df['bb_std'] = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        else:
            bbands = ta.bbands(df['close'], length=period, std=std)
            
            if bbands is not None:
                # Try different column name formats
                cols = bbands.columns.tolist()
                
                # Find the correct column names
                upper_col = [c for c in cols if 'BBU' in c or 'upper' in c.lower()]
                middle_col = [c for c in cols if 'BBM' in c or 'middle' in c.lower()]
                lower_col = [c for c in cols if 'BBL' in c or 'lower' in c.lower()]
                
                if upper_col and middle_col and lower_col:
                    df['bb_upper'] = bbands[upper_col[0]]
                    df['bb_middle'] = bbands[middle_col[0]]
                    df['bb_lower'] = bbands[lower_col[0]]
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                else:
                    # Fallback to manual calculation
                    df['bb_middle'] = df['close'].rolling(window=period).mean()
                    df['bb_std'] = df['close'].rolling(window=period).std()
                    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std)
                    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std)
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual stochastic calculation
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        else:
            stoch = ta.stoch(
                df['high'], 
                df['low'], 
                df['close'],
                k=k_period,
                d=d_period
            )
            
            if stoch is not None:
                cols = stoch.columns.tolist()
                k_col = [c for c in cols if 'STOCHk' in c or 'K' in c]
                d_col = [c for c in cols if 'STOCHd' in c or 'D' in c]
                
                if k_col and d_col:
                    df['stoch_k'] = stoch[k_col[0]]
                    df['stoch_d'] = stoch[d_col[0]]
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=period).mean()
        else:
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=period)
            if isinstance(atr_result, pd.DataFrame):
                cols = [c for c in atr_result.columns if 'ATR' in c]
                df['atr'] = atr_result[cols[0]] if cols else atr_result.iloc[:, 0]
            else:
                df['atr'] = atr_result
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        df = df.copy()
        
        if not HAS_PANDAS_TA:
            # Manual calculations
            # On-Balance Volume
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Volume Moving Average
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Money Flow Index (simplified)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            df['mfi'] = mfi
        else:
            # On-Balance Volume
            obv_result = ta.obv(df['close'], df['volume'])
            df['obv'] = obv_result if isinstance(obv_result, pd.Series) else obv_result.iloc[:, 0]
            
            # Volume Moving Average
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            
            # Money Flow Index
            mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            if mfi is not None:
                if isinstance(mfi, pd.DataFrame):
                    cols = [c for c in mfi.columns if 'MFI' in c]
                    df['mfi'] = mfi[cols[0]] if cols else mfi.iloc[:, 0]
                else:
                    df['mfi'] = mfi
        
        return df
    
    @staticmethod
    def identify_support_resistance(
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify support and resistance levels
        
        Args:
            df: DataFrame with price data
            window: Window size for identifying levels
        
        Returns:
            Tuple of (support levels, resistance levels)
        """
        df = df.copy()
        
        # Rolling min/max
        support = df['low'].rolling(window=window, center=True).min()
        resistance = df['high'].rolling(window=window, center=True).max()
        
        return support, resistance
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pivot points"""
        df = df.copy()
        
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        return df


if __name__ == "__main__":
    # Test indicators
    from data_handler import DataHandler
    
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2023-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        
        print("\nDataFrame with indicators:")
        print(df_with_indicators[['close', 'sma_20', 'rsi', 'macd', 'bb_upper']].tail())
