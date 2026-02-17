"""
Stochastic Process Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from strategies import BaseStrategy
from indicators import TechnicalIndicators

# Optional imports
try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    """
    Ornstein-Uhlenbeck Mean Reversion Strategy
    
    Models price as mean-reverting stochastic process:
    dX = θ(μ - X)dt + σdW
    
    Where:
    - θ: speed of mean reversion
    - μ: long-term mean
    - σ: volatility
    """
    
    def __init__(self, window=60, entry_threshold=1.5, exit_threshold=0.5):
        super().__init__("Ornstein-Uhlenbeck Mean Reversion")
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on OU process
        
        BUY: Price deviates below mean by entry_threshold * std
        SELL: Price reverts to mean (within exit_threshold * std)
        """
        df = df.copy()
        
        # Calculate log prices
        df['log_price'] = np.log(df['close'])
        
        # Calculate rolling statistics for OU process
        df['ou_mean'] = df['log_price'].rolling(window=self.window).mean()
        df['ou_std'] = df['log_price'].rolling(window=self.window).std()
        
        # Calculate deviation from mean (in standard deviations)
        df['ou_z_score'] = (df['log_price'] - df['ou_mean']) / df['ou_std']
        
        # Estimate mean reversion speed (half-life)
        df['half_life'] = self._estimate_half_life(df['log_price'], self.window)
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when significantly below mean
        df.loc[df['ou_z_score'] < -self.entry_threshold, 'signal'] = 1
        
        # Sell when price reverts to mean
        df.loc[df['ou_z_score'] > self.exit_threshold, 'signal'] = -1
        
        # Hold position until mean reversion
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using OU mean reversion")
        
        return df
    
    def _estimate_half_life(self, series: pd.Series, window: int) -> pd.Series:
        """
        Estimate half-life of mean reversion
        Half-life = -log(2) / θ
        """
        half_lives = []
        
        for i in range(len(series)):
            if i < window:
                half_lives.append(np.nan)
                continue
            
            y = series.iloc[i-window:i].values
            y_lag = y[:-1]  # All but last
            y_diff = np.diff(y)  # Differences
            
            # Simple linear regression: dy = θ*y_lag
            if len(y_lag) > 0 and len(y_diff) > 0 and np.std(y_lag) > 0:
                try:
                    # Calculate theta using covariance
                    covariance = np.mean((y_diff - np.mean(y_diff)) * (y_lag - np.mean(y_lag)))
                    variance = np.var(y_lag)
                    
                    if variance > 0:
                        theta = covariance / variance
                        
                        if theta < 0:
                            half_life = -np.log(2) / theta
                        else:
                            half_life = np.inf
                        
                        half_lives.append(half_life)
                    else:
                        half_lives.append(np.nan)
                except:
                    half_lives.append(np.nan)
            else:
                half_lives.append(np.nan)
        
        return pd.Series(half_lives, index=series.index)


class KalmanFilterStrategy(BaseStrategy):
    """
    Kalman Filter Strategy
    
    Uses Kalman filter to estimate true price and trade deviations
    Treats observed price as noisy measurement of true price
    """
    
    def __init__(self, delta=1e-5, transition_covariance=1e-4):
        super().__init__("Kalman Filter")
        self.delta = delta  # Transition covariance scaling
        self.transition_covariance = transition_covariance
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Kalman filter estimates
        
        BUY: Actual price below Kalman estimate
        SELL: Actual price above Kalman estimate
        """
        df = df.copy()
        
        # Apply Kalman filter
        df['kalman_mean'], df['kalman_std'] = self._kalman_filter(df['close'])
        
        # Calculate deviation
        df['kalman_deviation'] = (df['close'] - df['kalman_mean']) / df['kalman_std']
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when price is below Kalman estimate
        df.loc[df['kalman_deviation'] < -1.0, 'signal'] = 1
        
        # Sell when price is above Kalman estimate
        df.loc[df['kalman_deviation'] > 1.0, 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using Kalman filter")
        
        return df
    
    def _kalman_filter(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Kalman filter to price series
        
        Returns estimated mean and standard deviation
        """
        n = len(prices)
        
        # Initialize
        posterior_mean = prices.iloc[0]
        posterior_var = 1.0
        
        means = []
        stds = []
        
        for price in prices:
            # Prediction step
            prior_mean = posterior_mean
            prior_var = posterior_var + self.transition_covariance
            
            # Update step
            observation_var = self.delta
            kalman_gain = prior_var / (prior_var + observation_var)
            
            posterior_mean = prior_mean + kalman_gain * (price - prior_mean)
            posterior_var = (1 - kalman_gain) * prior_var
            
            means.append(posterior_mean)
            stds.append(np.sqrt(posterior_var))
        
        return pd.Series(means, index=prices.index), pd.Series(stds, index=prices.index)


class VolatilityClusteringStrategy(BaseStrategy):
    """
    GARCH-inspired Volatility Clustering Strategy
    
    Trades based on volatility regimes:
    - High volatility: mean reversion
    - Low volatility: trend following
    """
    
    def __init__(self, window=20, vol_threshold=1.5):
        super().__init__("Volatility Clustering")
        self.window = window
        self.vol_threshold = vol_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on volatility regimes
        
        High volatility regime: Mean reversion
        Low volatility regime: Trend following
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate realized volatility
        df['volatility'] = df['returns'].rolling(window=self.window).std()
        df['vol_mean'] = df['volatility'].rolling(window=self.window*2).mean()
        df['vol_std'] = df['volatility'].rolling(window=self.window*2).std()
        
        # Volatility z-score
        df['vol_zscore'] = (df['volatility'] - df['vol_mean']) / df['vol_std']
        
        # Identify regime
        df['high_vol_regime'] = df['vol_zscore'] > self.vol_threshold
        df['low_vol_regime'] = df['vol_zscore'] < -self.vol_threshold
        
        # Add technical indicators
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_moving_averages(df, [20, 50])
        
        # Generate signals based on regime
        df['signal'] = 0
        
        # High volatility: Mean reversion (use RSI)
        df.loc[df['high_vol_regime'] & (df['rsi'] < 30), 'signal'] = 1
        df.loc[df['high_vol_regime'] & (df['rsi'] > 70), 'signal'] = -1
        
        # Low volatility: Trend following (use MA crossover)
        df.loc[df['low_vol_regime'] & (df['sma_20'] > df['sma_50']), 'signal'] = 1
        df.loc[df['low_vol_regime'] & (df['sma_20'] < df['sma_50']), 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using volatility clustering")
        
        return df


class MomentumReversalStrategy(BaseStrategy):
    """
    Stochastic Momentum with Reversal
    
    Combines short-term momentum with mean reversion
    Uses stochastic calculus concepts
    """
    
    def __init__(self, momentum_window=10, reversion_window=60):
        super().__init__("Momentum-Reversal")
        self.momentum_window = momentum_window
        self.reversion_window = reversion_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals combining momentum and mean reversion
        
        Logic:
        - Short-term momentum (10 days)
        - Long-term mean reversion (60 days)
        - Trade when both align
        """
        df = df.copy()
        
        # Short-term momentum
        df['momentum'] = df['close'].pct_change(self.momentum_window)
        df['momentum_zscore'] = (df['momentum'] - df['momentum'].rolling(60).mean()) / df['momentum'].rolling(60).std()
        
        # Long-term mean reversion
        df['long_ma'] = df['close'].rolling(window=self.reversion_window).mean()
        df['long_std'] = df['close'].rolling(window=self.reversion_window).std()
        df['reversion_zscore'] = (df['close'] - df['long_ma']) / df['long_std']
        
        # Drift and diffusion estimates
        df['drift'] = df['close'].pct_change().rolling(20).mean()
        df['diffusion'] = df['close'].pct_change().rolling(20).std()
        
        # Generate signals
        df['signal'] = 0
        
        # Buy: Positive momentum + oversold on mean reversion
        df.loc[(df['momentum_zscore'] > 0.5) & (df['reversion_zscore'] < -1.0), 'signal'] = 1
        
        # Sell: Negative momentum + overbought on mean reversion
        df.loc[(df['momentum_zscore'] < -0.5) & (df['reversion_zscore'] > 1.0), 'signal'] = -1
        
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using momentum-reversal")
        
        return df


class JumpDiffusionStrategy(BaseStrategy):
    """
    Jump Diffusion Strategy
    
    Detects price jumps (sudden large moves) and trades reversals
    Based on Merton's jump diffusion model
    """
    
    def __init__(self, window=20, jump_threshold=3.0):
        super().__init__("Jump Diffusion")
        self.window = window
        self.jump_threshold = jump_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on jump detection
        
        BUY: After significant downward jump (oversold)
        SELL: After significant upward jump (overbought)
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Estimate continuous component (diffusion)
        df['returns_smooth'] = df['returns'].rolling(window=5).mean()
        df['vol'] = df['returns'].rolling(window=self.window).std()
        
        # Detect jumps (returns that are too large for normal diffusion)
        df['jump_indicator'] = np.abs(df['returns'] - df['returns_smooth']) / df['vol']
        df['is_jump'] = df['jump_indicator'] > self.jump_threshold
        
        # Classify jump direction
        df['jump_up'] = df['is_jump'] & (df['returns'] > 0)
        df['jump_down'] = df['is_jump'] & (df['returns'] < 0)
        
        # Generate signals (fade the jumps - mean reversion after jump)
        df['signal'] = 0
        
        # Buy after downward jump
        df.loc[df['jump_down'], 'signal'] = 1
        
        # Sell after upward jump
        df.loc[df['jump_up'], 'signal'] = -1
        
        # Exit after recovery (returns to pre-jump level)
        df['position'] = 0
        
        # Simple position management
        for i in range(1, len(df)):
            if df['signal'].iloc[i] != 0:
                df.loc[df.index[i], 'position'] = df['signal'].iloc[i]
            elif df['position'].iloc[i-1] != 0:
                # Check if recovered
                if df['position'].iloc[i-1] == 1 and df['returns'].iloc[i] > 0.01:
                    df.loc[df.index[i], 'position'] = 0  # Exit long
                elif df['position'].iloc[i-1] == -1 and df['returns'].iloc[i] < -0.01:
                    df.loc[df.index[i], 'position'] = 0  # Exit short
                else:
                    df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using jump diffusion")
        
        return df


if __name__ == "__main__":
    # Test stochastic strategies
    from data_handler import DataHandler
    
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2022-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        
        print("\nTesting Ornstein-Uhlenbeck Strategy...")
        ou_strategy = OrnsteinUhlenbeckStrategy()
        df_ou = ou_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_ou['signal'] != 0).sum()}")
        
        print("\nTesting Kalman Filter Strategy...")
        kalman_strategy = KalmanFilterStrategy()
        df_kalman = kalman_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_kalman['signal'] != 0).sum()}")
        
        print("\nTesting Volatility Clustering Strategy...")
        vol_strategy = VolatilityClusteringStrategy()
        df_vol = vol_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_vol['signal'] != 0).sum()}")
        
        print("\nTesting Jump Diffusion Strategy...")
        jump_strategy = JumpDiffusionStrategy()
        df_jump = jump_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_jump['signal'] != 0).sum()}")
