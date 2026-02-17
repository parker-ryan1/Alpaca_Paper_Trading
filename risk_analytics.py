"""
Risk Analytics Module
Advanced risk metrics and analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    Risk Analytics Engine
    
    Features:
    - VaR (Value at Risk)
    - CVaR (Conditional VaR)
    - Beta calculation
    - Correlation analysis
    - Stress testing
    - Risk attribution
    """
    
    def __init__(self):
        self.returns = None
        self.benchmark_returns = None
    
    def set_returns(self, returns: pd.Series):
        """Set portfolio returns"""
        self.returns = returns
    
    def set_benchmark(self, benchmark_returns: pd.Series):
        """Set benchmark returns"""
        self.benchmark_returns = benchmark_returns
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (default 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical VaR
            var = np.percentile(self.returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mu = self.returns.mean()
            sigma = self.returns.std()
            var = stats.norm.ppf(1 - confidence_level, mu, sigma)
        
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mu = self.returns.mean()
            sigma = self.returns.std()
            simulations = np.random.normal(mu, sigma, 10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Average of losses beyond VaR
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
        
        var = self.calculate_var(confidence_level)
        cvar = self.returns[self.returns <= var].mean()
        
        return cvar
    
    def calculate_beta(self) -> float:
        """
        Calculate portfolio beta relative to benchmark
        
        Beta = Cov(portfolio, benchmark) / Var(benchmark)
        """
        if self.returns is None or self.benchmark_returns is None:
            return 1.0
        
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 1.0
        
        covariance = aligned['portfolio'].cov(aligned['benchmark'])
        benchmark_variance = aligned['benchmark'].var()
        
        if benchmark_variance == 0:
            return 1.0
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_alpha(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Jensen's Alpha
        
        Alpha = Portfolio Return - (Rf + Beta * (Benchmark Return - Rf))
        """
        if self.returns is None or self.benchmark_returns is None:
            return 0.0
        
        beta = self.calculate_beta()
        
        portfolio_return = self.returns.mean() * 252  # Annualized
        benchmark_return = self.benchmark_returns.mean() * 252  # Annualized
        
        expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
        alpha = portfolio_return - expected_return
        
        return alpha
    
    def calculate_information_ratio(self) -> float:
        """
        Calculate Information Ratio
        
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        """
        if self.returns is None or self.benchmark_returns is None:
            return 0.0
        
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        excess_returns = aligned['portfolio'] - aligned['benchmark']
        
        if excess_returns.std() == 0:
            return 0.0
        
        ir = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return ir
    
    def calculate_tracking_error(self) -> float:
        """Calculate tracking error relative to benchmark"""
        if self.returns is None or self.benchmark_returns is None:
            return 0.0
        
        aligned = pd.DataFrame({
            'portfolio': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        excess_returns = aligned['portfolio'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return tracking_error
    
    def calculate_downside_deviation(
        self,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate downside deviation
        
        Only considers returns below target
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
        
        downside_returns = self.returns[self.returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        return downside_deviation
    
    def calculate_sortino_ratio(
        self,
        target_return: float = 0.0,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino Ratio
        
        Like Sharpe but only considers downside volatility
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
        
        excess_return = self.returns.mean() * 252 - risk_free_rate
        downside_dev = self.calculate_downside_deviation(target_return / 252)
        
        if downside_dev == 0:
            return 0.0
        
        sortino = excess_return / downside_dev
        
        return sortino
    
    def calculate_calmar_ratio(self, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio
        
        Calmar = Annualized Return / Max Drawdown
        """
        if self.returns is None or len(self.returns) == 0 or max_drawdown == 0:
            return 0.0
        
        annualized_return = self.returns.mean() * 252
        calmar = annualized_return / abs(max_drawdown / 100)
        
        return calmar
    
    def calculate_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        df = pd.DataFrame(returns_dict)
        correlation = df.corr()
        
        return correlation
    
    def stress_test(
        self,
        scenario: str = 'market_crash'
    ) -> Dict:
        """
        Run stress test scenarios
        
        Scenarios:
        - market_crash: -20% market drop
        - volatility_spike: 3x volatility
        - interest_rate_shock: +200bps
        """
        if self.returns is None or len(self.returns) == 0:
            return {}
        
        results = {}
        
        if scenario == 'market_crash':
            # Simulate -20% market drop
            shocked_return = self.returns.mean() * 0.8 - 0.20
            results['expected_loss'] = shocked_return
            results['portfolio_impact'] = shocked_return * 100
        
        elif scenario == 'volatility_spike':
            # Simulate 3x volatility
            current_vol = self.returns.std() * np.sqrt(252)
            shocked_vol = current_vol * 3
            results['current_volatility'] = current_vol
            results['shocked_volatility'] = shocked_vol
            results['var_95'] = self.calculate_var(0.95) * 3
        
        elif scenario == 'interest_rate_shock':
            # Simulate +200bps rate increase
            # Simplified: assume duration-based impact
            results['rate_increase'] = 0.02
            results['estimated_impact'] = -0.05  # -5% portfolio impact
        
        return results
    
    def get_risk_metrics_summary(
        self,
        max_drawdown: float = 0.0,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """Get comprehensive risk metrics summary"""
        
        metrics = {
            'var_95': self.calculate_var(0.95),
            'cvar_95': self.calculate_cvar(0.95),
            'volatility': self.returns.std() * np.sqrt(252) if self.returns is not None else 0,
            'downside_deviation': self.calculate_downside_deviation(),
            'sortino_ratio': self.calculate_sortino_ratio(risk_free_rate=risk_free_rate),
        }
        
        if self.benchmark_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(),
                'alpha': self.calculate_alpha(risk_free_rate),
                'information_ratio': self.calculate_information_ratio(),
                'tracking_error': self.calculate_tracking_error(),
            })
        
        if max_drawdown != 0:
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(max_drawdown)
        
        return metrics
    
    def generate_risk_report(
        self,
        max_drawdown: float = 0.0,
        risk_free_rate: float = 0.02
    ) -> str:
        """Generate formatted risk report"""
        
        metrics = self.get_risk_metrics_summary(max_drawdown, risk_free_rate)
        
        report = """
╔═══════════════════════════════════════════════════════════╗
║               COMPREHENSIVE RISK ANALYSIS                 ║
╚═══════════════════════════════════════════════════════════╝

VALUE AT RISK (VaR)
-------------------
VaR (95%):                {var_95:.2%}
CVaR (95%):               {cvar_95:.2%}

VOLATILITY METRICS
------------------
Annual Volatility:        {volatility:.2%}
Downside Deviation:       {downside_deviation:.2%}

RISK-ADJUSTED RETURNS
---------------------
Sortino Ratio:            {sortino_ratio:.2f}
""".format(**metrics)
        
        if 'beta' in metrics:
            report += """
MARKET RISK
-----------
Beta:                     {beta:.2f}
Alpha:                    {alpha:.2%}
Information Ratio:        {information_ratio:.2f}
Tracking Error:           {tracking_error:.2%}
""".format(**metrics)
        
        if 'calmar_ratio' in metrics:
            report += """
DRAWDOWN METRICS
----------------
Calmar Ratio:             {calmar_ratio:.2f}
Max Drawdown:             {max_dd:.2%}
""".format(calmar_ratio=metrics['calmar_ratio'], max_dd=max_drawdown/100)
        
        return report


if __name__ == "__main__":
    # Test risk analyzer
    np.random.seed(42)
    
    # Generate sample returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
    
    # Initialize analyzer
    analyzer = RiskAnalyzer()
    analyzer.set_returns(returns)
    analyzer.set_benchmark(benchmark_returns)
    
    # Print risk report
    print(analyzer.generate_risk_report(max_drawdown=-5.2))
    
    # Stress test
    print("\nStress Test (Market Crash):")
    crash_results = analyzer.stress_test('market_crash')
    for key, value in crash_results.items():
        print(f"  {key}: {value}")
