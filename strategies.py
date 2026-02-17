"""
Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from indicators import TechnicalIndicators
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.positions = {}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError("Subclass must implement generate_signals()")
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05
    ) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Available capital
            price: Current price
            risk_per_trade: Risk per trade as decimal (0.02 = 2%)
            stop_loss_pct: Stop loss percentage
        
        Returns:
            Number of shares to buy
        """
        risk_amount = capital * risk_per_trade
        shares = int(risk_amount / (price * stop_loss_pct))
        return max(shares, 0)


class MACDStrategy(BaseStrategy):
    """MACD Crossover Strategy"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__("MACD Crossover")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MACD crossovers
        
        Signal Logic:
        - BUY: MACD crosses above signal line
        - SELL: MACD crosses below signal line
        """
        df = df.copy()
        
        # Add MACD if not present
        if 'macd' not in df.columns:
            df = TechnicalIndicators.add_macd(df, self.fast, self.slow, self.signal)
        
        # Generate signals
        df['signal'] = 0
        
        # MACD crosses above signal line
        df.loc[(df['macd'] > df['macd_signal']) & 
               (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'signal'] = 1
        
        # MACD crosses below signal line
        df.loc[(df['macd'] < df['macd_signal']) & 
               (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'signal'] = -1
        
        # Create position column (1 = long, 0 = flat, -1 = short)
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using MACD strategy")
        
        return df


class RSIMeanReversion(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on RSI oversold/overbought
        
        Signal Logic:
        - BUY: RSI crosses above oversold level
        - SELL: RSI crosses below overbought level
        """
        df = df.copy()
        
        # Add RSI if not present
        if 'rsi' not in df.columns:
            df = TechnicalIndicators.add_rsi(df, self.rsi_period)
        
        df['signal'] = 0
        
        # Buy when RSI crosses above oversold
        df.loc[(df['rsi'] > self.oversold) & 
               (df['rsi'].shift(1) <= self.oversold), 'signal'] = 1
        
        # Sell when RSI crosses below overbought
        df.loc[(df['rsi'] < self.overbought) & 
               (df['rsi'].shift(1) >= self.overbought), 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using RSI strategy")
        
        return df


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, period=20, std=2):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std = std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Bollinger Bands
        
        Signal Logic:
        - BUY: Price touches lower band
        - SELL: Price touches upper band
        """
        df = df.copy()
        
        # Add Bollinger Bands if not present
        if 'bb_upper' not in df.columns:
            df = TechnicalIndicators.add_bollinger_bands(df, self.period, self.std)
        
        df['signal'] = 0
        
        # Buy when price touches lower band
        df.loc[df['close'] <= df['bb_lower'], 'signal'] = 1
        
        # Sell when price touches upper band
        df.loc[df['close'] >= df['bb_upper'], 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using BB strategy")
        
        return df


class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, fast_period=50, slow_period=200):
        super().__init__("MA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MA crossovers
        
        Signal Logic:
        - BUY: Fast MA crosses above Slow MA (Golden Cross)
        - SELL: Fast MA crosses below Slow MA (Death Cross)
        """
        df = df.copy()
        
        # Add MAs if not present
        if f'sma_{self.fast_period}' not in df.columns:
            df = TechnicalIndicators.add_moving_averages(
                df, 
                [self.fast_period, self.slow_period]
            )
        
        fast_ma = f'sma_{self.fast_period}'
        slow_ma = f'sma_{self.slow_period}'
        
        df['signal'] = 0
        
        # Golden Cross
        df.loc[(df[fast_ma] > df[slow_ma]) & 
               (df[fast_ma].shift(1) <= df[slow_ma].shift(1)), 'signal'] = 1
        
        # Death Cross
        df.loc[(df[fast_ma] < df[slow_ma]) & 
               (df[fast_ma].shift(1) >= df[slow_ma].shift(1)), 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using MA crossover")
        
        return df


class MultiFactorStrategy(BaseStrategy):
    """
    Multi-Factor Strategy combining multiple indicators
    """
    
    def __init__(self):
        super().__init__("Multi-Factor")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on multiple factors
        
        Signal Logic:
        - BUY: When majority of indicators are bullish
        - SELL: When majority of indicators are bearish
        """
        df = df.copy()
        
        # Ensure all indicators are present
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Initialize scoring system
        df['score'] = 0
        
        # MACD Factor
        df.loc[df['macd'] > df['macd_signal'], 'score'] += 1
        df.loc[df['macd'] < df['macd_signal'], 'score'] -= 1
        
        # RSI Factor
        df.loc[df['rsi'] < 30, 'score'] += 1  # Oversold
        df.loc[df['rsi'] > 70, 'score'] -= 1  # Overbought
        
        # MA Factor
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df.loc[df['sma_50'] > df['sma_200'], 'score'] += 1
            df.loc[df['sma_50'] < df['sma_200'], 'score'] -= 1
        
        # Price vs BB Factor
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            df.loc[df['close'] < df['bb_lower'], 'score'] += 1
            df.loc[df['close'] > df['bb_upper'], 'score'] -= 1
        
        # Generate signals based on score
        df['signal'] = 0
        df.loc[df['score'] >= 2, 'signal'] = 1   # Bullish
        df.loc[df['score'] <= -2, 'signal'] = -1  # Bearish
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using multi-factor")
        
        return df


# Import advanced strategies
try:
    from ml_strategies import (
        RandomForestStrategy,
        GradientBoostingStrategy,
        EnsembleMLStrategy
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML strategies not available. Install scikit-learn to enable.")

try:
    from stochastic_strategies import (
        OrnsteinUhlenbeckStrategy,
        KalmanFilterStrategy,
        VolatilityClusteringStrategy,
        MomentumReversalStrategy,
        JumpDiffusionStrategy
    )
    STOCHASTIC_AVAILABLE = True
except ImportError:
    STOCHASTIC_AVAILABLE = False
    logger.warning("Stochastic strategies not available.")


# Strategy Factory
STRATEGIES = {
    # Classic Technical Strategies
    'macd': MACDStrategy,
    'rsi': RSIMeanReversion,
    'bollinger': BollingerBandsStrategy,
    'ma_crossover': MovingAverageCrossover,
    'multi_factor': MultiFactorStrategy,
}

# Add ML strategies if available
if ML_AVAILABLE:
    STRATEGIES.update({
        'random_forest': RandomForestStrategy,
        'gradient_boosting': GradientBoostingStrategy,
        'ensemble_ml': EnsembleMLStrategy,
    })

# Add Stochastic strategies if available
if STOCHASTIC_AVAILABLE:
    STRATEGIES.update({
        'ornstein_uhlenbeck': OrnsteinUhlenbeckStrategy,
        'kalman_filter': KalmanFilterStrategy,
        'volatility_clustering': VolatilityClusteringStrategy,
        'momentum_reversal': MomentumReversalStrategy,
        'jump_diffusion': JumpDiffusionStrategy,
    })


def get_strategy(strategy_name: str, **kwargs):
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return STRATEGIES[strategy_name](**kwargs)


if __name__ == "__main__":
    # Test strategies
    from data_handler import DataHandler
    
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2023-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        
        # Test MACD strategy
        strategy = MACDStrategy()
        df_with_signals = strategy.generate_signals(df)
        
        print("\nSignals generated:")
        print(df_with_signals[df_with_signals['signal'] != 0][['close', 'macd', 'macd_signal', 'signal']])
