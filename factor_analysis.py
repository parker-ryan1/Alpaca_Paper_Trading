"""
Factor Analysis Module
Multi-factor models and factor attribution
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorAnalyzer:
    """
    Factor Analysis for Trading Strategies
    
    Features:
    - Fama-French 3-factor model
    - Carhart 4-factor model
    - Factor exposure analysis
    - Factor attribution
    - Style analysis
    """
    
    def __init__(self):
        self.returns = None
        self.factors = None
    
    def set_returns(self, returns: pd.Series):
        """Set strategy returns"""
        self.returns = returns
    
    def set_factors(self, factors: pd.DataFrame):
        """
        Set factor returns
        
        Expected columns:
        - Mkt-RF: Market excess return
        - SMB: Small Minus Big (size factor)
        - HML: High Minus Low (value factor)
        - MOM: Momentum factor (optional)
        - RF: Risk-free rate
        """
        self.factors = factors
    
    def generate_mock_factors(self, periods: int = 252) -> pd.DataFrame:
        """
        Generate mock factor returns for testing
        
        Real implementation would fetch from Kenneth French Data Library
        """
        np.random.seed(42)
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='D')
        
        factors = pd.DataFrame({
            'Mkt-RF': np.random.normal(0.0005, 0.01, periods),  # Market excess return
            'SMB': np.random.normal(0.0002, 0.005, periods),    # Size factor
            'HML': np.random.normal(0.0001, 0.005, periods),    # Value factor
            'MOM': np.random.normal(0.0003, 0.006, periods),    # Momentum factor
            'RF': np.full(periods, 0.0001)                      # Risk-free rate
        }, index=dates)
        
        return factors
    
    def run_fama_french_3factor(self) -> Dict:
        """
        Run Fama-French 3-factor regression
        
        R - RF = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε
        """
        if self.returns is None:
            logger.error("Returns not set")
            return {}
        
        if self.factors is None:
            logger.info("Generating mock factors")
            self.factors = self.generate_mock_factors(len(self.returns))
        
        # Align returns and factors
        aligned = pd.DataFrame({
            'returns': self.returns,
            **self.factors
        }).dropna()
        
        if len(aligned) < 30:
            logger.warning("Insufficient data for regression")
            return {}
        
        # Calculate excess returns
        aligned['excess_return'] = aligned['returns'] - aligned['RF']
        
        # Prepare regression
        X = aligned[['Mkt-RF', 'SMB', 'HML']]
        y = aligned['excess_return']
        
        # Add constant for alpha
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        alpha = beta[0]
        beta_mkt = beta[1]
        beta_smb = beta[2]
        beta_hml = beta[3]
        
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2) if len(residuals) > 0 else np.sum((y - X @ beta) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # T-statistics
        mse = ss_res / (len(y) - len(beta))
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        t_stats = beta / np.sqrt(var_beta)
        
        return {
            'model': 'Fama-French 3-Factor',
            'alpha': alpha * 252,  # Annualized
            'beta_market': beta_mkt,
            'beta_size': beta_smb,
            'beta_value': beta_hml,
            'r_squared': r_squared,
            't_stat_alpha': t_stats[0],
            't_stat_market': t_stats[1],
            't_stat_size': t_stats[2],
            't_stat_value': t_stats[3]
        }
    
    def run_carhart_4factor(self) -> Dict:
        """
        Run Carhart 4-factor regression (adds momentum)
        
        R - RF = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(MOM) + ε
        """
        if self.returns is None or self.factors is None:
            logger.error("Returns or factors not set")
            return {}
        
        # Align returns and factors
        aligned = pd.DataFrame({
            'returns': self.returns,
            **self.factors
        }).dropna()
        
        if len(aligned) < 30:
            return {}
        
        # Calculate excess returns
        aligned['excess_return'] = aligned['returns'] - aligned['RF']
        
        # Prepare regression with momentum
        X = aligned[['Mkt-RF', 'SMB', 'HML', 'MOM']]
        y = aligned['excess_return']
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        alpha = beta[0]
        beta_mkt = beta[1]
        beta_smb = beta[2]
        beta_hml = beta[3]
        beta_mom = beta[4]
        
        # R-squared
        ss_res = np.sum(residuals ** 2) if len(residuals) > 0 else np.sum((y - X @ beta) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'model': 'Carhart 4-Factor',
            'alpha': alpha * 252,
            'beta_market': beta_mkt,
            'beta_size': beta_smb,
            'beta_value': beta_hml,
            'beta_momentum': beta_mom,
            'r_squared': r_squared
        }
    
    def calculate_factor_exposure(self) -> pd.DataFrame:
        """Calculate factor exposures over time"""
        if self.returns is None or self.factors is None:
            return pd.DataFrame()
        
        # Rolling regression for factor exposures
        window = 60  # 60-day rolling window
        
        exposures = []
        
        for i in range(window, len(self.returns)):
            subset_returns = self.returns.iloc[i-window:i]
            subset_factors = self.factors.iloc[i-window:i]
            
            # Run mini regression
            aligned = pd.DataFrame({
                'returns': subset_returns,
                **subset_factors
            }).dropna()
            
            if len(aligned) < 30:
                continue
            
            aligned['excess_return'] = aligned['returns'] - aligned['RF']
            
            X = aligned[['Mkt-RF', 'SMB', 'HML']]
            y = aligned['excess_return']
            X = np.column_stack([np.ones(len(X)), X])
            
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            exposures.append({
                'date': subset_returns.index[-1],
                'market': beta[1],
                'size': beta[2],
                'value': beta[3]
            })
        
        return pd.DataFrame(exposures).set_index('date')
    
    def decompose_returns(self) -> Dict:
        """
        Decompose returns into factor contributions
        
        Return = Alpha + Factor Contributions + Residual
        """
        results = self.run_fama_french_3factor()
        
        if not results:
            return {}
        
        if self.factors is None:
            return {}
        
        # Calculate factor contributions
        aligned = pd.DataFrame({
            'returns': self.returns,
            **self.factors
        }).dropna()
        
        aligned['excess_return'] = aligned['returns'] - aligned['RF']
        
        # Calculate contributions
        market_contrib = results['beta_market'] * aligned['Mkt-RF'].mean() * 252
        size_contrib = results['beta_size'] * aligned['SMB'].mean() * 252
        value_contrib = results['beta_value'] * aligned['HML'].mean() * 252
        
        total_return = self.returns.mean() * 252
        alpha = results['alpha']
        residual = total_return - (alpha + market_contrib + size_contrib + value_contrib)
        
        return {
            'total_return': total_return,
            'alpha': alpha,
            'market_contribution': market_contrib,
            'size_contribution': size_contrib,
            'value_contribution': value_contrib,
            'residual': residual
        }
    
    def style_analysis(self) -> Dict:
        """
        Determine investment style based on factor exposures
        
        Styles:
        - Growth vs Value (HML factor)
        - Large cap vs Small cap (SMB factor)
        - Momentum vs Contrarian (MOM factor)
        """
        results = self.run_fama_french_3factor()
        
        if not results:
            return {}
        
        # Classify style
        styles = []
        
        # Value vs Growth
        if results['beta_value'] > 0.2:
            styles.append('Value')
        elif results['beta_value'] < -0.2:
            styles.append('Growth')
        else:
            styles.append('Blend')
        
        # Size
        if results['beta_size'] > 0.2:
            styles.append('Small Cap')
        elif results['beta_size'] < -0.2:
            styles.append('Large Cap')
        else:
            styles.append('Mid Cap')
        
        # Market exposure
        if results['beta_market'] > 1.2:
            market_style = 'Aggressive'
        elif results['beta_market'] < 0.8:
            market_style = 'Defensive'
        else:
            market_style = 'Neutral'
        
        return {
            'style': ' / '.join(styles),
            'market_exposure': market_style,
            'value_tilt': results['beta_value'],
            'size_tilt': results['beta_size']
        }
    
    def generate_factor_report(self) -> str:
        """Generate comprehensive factor analysis report"""
        
        ff3 = self.run_fama_french_3factor()
        decomp = self.decompose_returns()
        style = self.style_analysis()
        
        if not ff3:
            return "Insufficient data for factor analysis"
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║              FACTOR ANALYSIS REPORT                       ║
╚═══════════════════════════════════════════════════════════╝

FAMA-FRENCH 3-FACTOR MODEL
--------------------------
Alpha (Annualized):       {ff3['alpha']:.2%}
R-Squared:                {ff3['r_squared']:.2%}

FACTOR EXPOSURES
----------------
Market Beta:              {ff3['beta_market']:.2f}
Size Factor (SMB):        {ff3['beta_size']:.2f}
Value Factor (HML):       {ff3['beta_value']:.2f}

STATISTICAL SIGNIFICANCE
------------------------
Alpha t-stat:             {ff3['t_stat_alpha']:.2f}
Market t-stat:            {ff3['t_stat_market']:.2f}
Size t-stat:              {ff3['t_stat_size']:.2f}
Value t-stat:             {ff3['t_stat_value']:.2f}
"""
        
        if decomp:
            report += f"""
RETURN DECOMPOSITION
--------------------
Total Return:             {decomp['total_return']:.2%}
Alpha:                    {decomp['alpha']:.2%}
Market Contribution:      {decomp['market_contribution']:.2%}
Size Contribution:        {decomp['size_contribution']:.2%}
Value Contribution:       {decomp['value_contribution']:.2%}
Residual:                 {decomp['residual']:.2%}
"""
        
        if style:
            report += f"""
INVESTMENT STYLE
----------------
Style:                    {style['style']}
Market Exposure:          {style['market_exposure']}
Value Tilt:               {style['value_tilt']:.2f}
Size Tilt:                {style['size_tilt']:.2f}
"""
        
        return report


if __name__ == "__main__":
    # Test factor analyzer
    np.random.seed(42)
    
    # Generate sample returns
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 252),
        index=pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    )
    
    # Initialize analyzer
    analyzer = FactorAnalyzer()
    analyzer.set_returns(returns)
    
    # Generate and print report
    print(analyzer.generate_factor_report())
    
    # Style analysis
    style = analyzer.style_analysis()
    print(f"\nInvestment Style: {style['style']}")
    print(f"Market Exposure: {style['market_exposure']}")
