"""
Machine Learning Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from strategies import BaseStrategy
from indicators import TechnicalIndicators

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. ML strategies won't work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestStrategy(BaseStrategy):
    """
    Random Forest ML Strategy
    Uses technical indicators as features to predict price direction
    """
    
    def __init__(self, n_estimators=100, lookback=50):
        super().__init__("Random Forest ML")
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using Random Forest classifier
        
        Features: RSI, MACD, BB position, Volume ratio, Price momentum
        Target: Next day price direction (up/down)
        """
        df = df.copy()
        
        # Add all indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Create features
        features = self._create_features(df)
        
        # Create target (next day return > 0)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        valid_data = df.dropna()
        
        if len(valid_data) < self.lookback:
            logger.warning("Not enough data for ML strategy")
            df['signal'] = 0
            df['position'] = 0
            return df
        
        # Split data
        split_idx = len(valid_data) - self.lookback
        train_data = valid_data.iloc[:split_idx]
        test_data = valid_data.iloc[split_idx:]
        
        if len(train_data) < 50:
            logger.warning("Not enough training data")
            df['signal'] = 0
            df['position'] = 0
            return df
        
        # Prepare training data
        feature_cols = [col for col in features if col in train_data.columns]
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Predict on all data
        X_all = valid_data[feature_cols]
        X_all_scaled = self.scaler.transform(X_all)
        predictions = self.model.predict(X_all_scaled)
        probabilities = self.model.predict_proba(X_all_scaled)
        
        # Generate signals based on predictions and confidence
        valid_data['prediction'] = predictions
        valid_data['confidence'] = probabilities[:, 1]  # Probability of up move
        
        # Signal: Buy if predict up with high confidence, Sell if predict down
        valid_data['signal'] = 0
        valid_data.loc[valid_data['confidence'] > 0.6, 'signal'] = 1  # High confidence up
        valid_data.loc[valid_data['confidence'] < 0.4, 'signal'] = -1  # High confidence down
        
        valid_data['position'] = valid_data['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Merge back to original df
        df.loc[valid_data.index, 'signal'] = valid_data['signal']
        df.loc[valid_data.index, 'position'] = valid_data['position']
        df['signal'] = df['signal'].fillna(0)
        df['position'] = df['position'].fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using Random Forest ML")
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> list:
        """Create feature list for ML model"""
        features = []
        
        # Price-based features
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['returns_5'] = df['close'].pct_change(5)
            df['returns_20'] = df['close'].pct_change(20)
            features.extend(['returns', 'returns_5', 'returns_20'])
        
        # Technical indicators
        if 'rsi' in df.columns:
            features.append('rsi')
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_diff'] = df['macd'] - df['macd_signal']
            features.extend(['macd', 'macd_diff'])
        
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features.append('bb_position')
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features.append('volume_ratio')
        
        # Moving averages
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_ratio'] = df['sma_20'] / df['sma_50']
            features.append('ma_ratio')
        
        # Volatility
        if 'atr' in df.columns:
            features.append('atr')
        
        return features


class GradientBoostingStrategy(BaseStrategy):
    """
    Gradient Boosting Strategy
    More powerful than Random Forest, uses boosting
    """
    
    def __init__(self, n_estimators=100, lookback=50):
        super().__init__("Gradient Boosting ML")
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using Gradient Boosting"""
        df = df.copy()
        
        # Add indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Create features
        features = self._create_features(df)
        
        # Create target (next day return > 0)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN
        valid_data = df.dropna()
        
        if len(valid_data) < self.lookback:
            df['signal'] = 0
            df['position'] = 0
            return df
        
        # Split
        split_idx = len(valid_data) - self.lookback
        train_data = valid_data.iloc[:split_idx]
        
        if len(train_data) < 50:
            df['signal'] = 0
            df['position'] = 0
            return df
        
        # Prepare data
        feature_cols = [col for col in features if col in train_data.columns]
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Predict
        X_all = valid_data[feature_cols]
        X_all_scaled = self.scaler.transform(X_all)
        predictions = self.model.predict(X_all_scaled)
        probabilities = self.model.predict_proba(X_all_scaled)
        
        # Generate signals
        valid_data['prediction'] = predictions
        valid_data['confidence'] = probabilities[:, 1]
        
        valid_data['signal'] = 0
        valid_data.loc[valid_data['confidence'] > 0.65, 'signal'] = 1
        valid_data.loc[valid_data['confidence'] < 0.35, 'signal'] = -1
        
        valid_data['position'] = valid_data['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Merge back
        df.loc[valid_data.index, 'signal'] = valid_data['signal']
        df.loc[valid_data.index, 'position'] = valid_data['position']
        df['signal'] = df['signal'].fillna(0)
        df['position'] = df['position'].fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using Gradient Boosting")
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> list:
        """Create feature list"""
        features = []
        
        # Price momentum
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['returns_5'] = df['close'].pct_change(5)
            df['returns_20'] = df['close'].pct_change(20)
            df['returns_std'] = df['returns'].rolling(20).std()
            features.extend(['returns', 'returns_5', 'returns_20', 'returns_std'])
        
        # Technical indicators
        if 'rsi' in df.columns:
            df['rsi_ma'] = df['rsi'].rolling(5).mean()
            features.extend(['rsi', 'rsi_ma'])
        
        if 'macd' in df.columns:
            df['macd_diff'] = df['macd'] - df['macd_signal']
            features.extend(['macd', 'macd_hist', 'macd_diff'])
        
        if 'bb_upper' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            features.extend(['bb_position', 'bb_width'])
        
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_std'] = df['volume'].rolling(20).std()
            features.extend(['volume_ratio', 'volume_std'])
        
        if 'atr' in df.columns:
            features.append('atr')
        
        return features


class EnsembleMLStrategy(BaseStrategy):
    """
    Ensemble ML Strategy
    Combines multiple ML models for more robust predictions
    """
    
    def __init__(self):
        super().__init__("Ensemble ML")
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")
        
        self.rf_strategy = RandomForestStrategy(n_estimators=50)
        self.gb_strategy = GradientBoostingStrategy(n_estimators=50)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals by combining multiple ML models
        Uses voting: majority decides the signal
        """
        df = df.copy()
        
        # Get signals from each model
        df_rf = self.rf_strategy.generate_signals(df.copy())
        df_gb = self.gb_strategy.generate_signals(df.copy())
        
        # Combine signals (voting)
        df['signal'] = 0
        
        # Both agree on buy
        df.loc[(df_rf['signal'] == 1) & (df_gb['signal'] == 1), 'signal'] = 1
        
        # Both agree on sell
        df.loc[(df_rf['signal'] == -1) & (df_gb['signal'] == -1), 'signal'] = -1
        
        # Strong agreement gets priority
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals using Ensemble ML")
        
        return df


if __name__ == "__main__":
    # Test ML strategies
    from data_handler import DataHandler
    
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2022-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        
        print("\nTesting Random Forest Strategy...")
        rf_strategy = RandomForestStrategy()
        df_rf = rf_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_rf['signal'] != 0).sum()}")
        
        print("\nTesting Gradient Boosting Strategy...")
        gb_strategy = GradientBoostingStrategy()
        df_gb = gb_strategy.generate_signals(df.copy())
        print(f"Signals: {(df_gb['signal'] != 0).sum()}")
        
        print("\nTesting Ensemble Strategy...")
        ensemble = EnsembleMLStrategy()
        df_ensemble = ensemble.generate_signals(df.copy())
        print(f"Signals: {(df_ensemble['signal'] != 0).sum()}")
