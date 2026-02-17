"""
Backtesting Engine
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Make quantstats optional
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.positions_history = []
        
    def run(
        self,
        df: pd.DataFrame,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        position_size_pct: float = 0.20
    ) -> pd.DataFrame:
        """
        Run backtest on DataFrame with signals
        
        Args:
            df: DataFrame with 'signal' column
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size as % of capital
        
        Returns:
            DataFrame with backtest results
        """
        df = df.copy()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        # Track portfolio metrics
        df['capital'] = np.nan
        df['holdings'] = np.nan
        df['total'] = np.nan
        df['returns'] = np.nan
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            date = df.index[i]
            
            # Calculate current portfolio value
            holdings_value = position * current_price
            total_value = capital + holdings_value
            
            df.loc[df.index[i], 'capital'] = capital
            df.loc[df.index[i], 'holdings'] = holdings_value
            df.loc[df.index[i], 'total'] = total_value
            
            # Check for stop loss or take profit
            if position > 0:
                price_change = (current_price - entry_price) / entry_price
                
                # Stop loss hit
                if price_change <= -stop_loss_pct:
                    logger.info(f"Stop loss hit at {date}: {price_change:.2%}")
                    capital, position = self._close_position(
                        capital, position, current_price, date, entry_date, entry_price, 'stop_loss'
                    )
                    entry_price = 0
                    entry_date = None
                    continue
                
                # Take profit hit
                if price_change >= take_profit_pct:
                    logger.info(f"Take profit hit at {date}: {price_change:.2%}")
                    capital, position = self._close_position(
                        capital, position, current_price, date, entry_date, entry_price, 'take_profit'
                    )
                    entry_price = 0
                    entry_date = None
                    continue
            
            # Process signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                max_position_value = total_value * position_size_pct
                shares_to_buy = int(max_position_value / current_price)
                
                if shares_to_buy > 0:
                    # Apply slippage and commission
                    actual_price = current_price * (1 + self.slippage)
                    cost = shares_to_buy * actual_price
                    commission_cost = cost * self.commission
                    
                    if capital >= cost + commission_cost:
                        capital -= (cost + commission_cost)
                        position = shares_to_buy
                        entry_price = actual_price
                        entry_date = date
                        
                        self.trades.append({
                            'date': date,
                            'type': 'BUY',
                            'price': actual_price,
                            'shares': shares_to_buy,
                            'value': cost,
                            'commission': commission_cost,
                            'capital': capital
                        })
                        
                        logger.info(f"BUY {shares_to_buy} shares at ${actual_price:.2f} on {date}")
            
            elif signal == -1 and position > 0:  # Sell signal
                capital, position = self._close_position(
                    capital, position, current_price, date, entry_date, entry_price, 'signal'
                )
                entry_price = 0
                entry_date = None
        
        # Close any remaining position
        if position > 0:
            final_price = df['close'].iloc[-1]
            capital, position = self._close_position(
                capital, position, final_price, df.index[-1], entry_date, entry_price, 'final'
            )
        
        # Calculate returns
        df['returns'] = df['total'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        
        # Store equity curve (include returns column)
        self.equity_curve = df[['total', 'returns', 'cumulative_returns']].copy()
        
        logger.info(f"\nBacktest complete. Trades: {len(self.trades)}")
        logger.info(f"Final capital: ${df['total'].iloc[-1]:,.2f}")
        logger.info(f"Total return: {(df['total'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%")
        
        return df
    
    def _close_position(
        self,
        capital: float,
        position: int,
        current_price: float,
        date,
        entry_date,
        entry_price: float,
        reason: str
    ) -> tuple:
        """Close an open position"""
        # Apply slippage and commission
        actual_price = current_price * (1 - self.slippage)
        proceeds = position * actual_price
        commission_cost = proceeds * self.commission
        
        capital += (proceeds - commission_cost)
        
        # Calculate P&L
        pnl = (actual_price - entry_price) * position - commission_cost
        pnl_pct = (actual_price / entry_price - 1) * 100 if entry_price > 0 else 0
        
        self.trades.append({
            'date': date,
            'type': 'SELL',
            'price': actual_price,
            'shares': position,
            'value': proceeds,
            'commission': commission_cost,
            'capital': capital,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_days': (date - entry_date).days if entry_date else 0
        })
        
        logger.info(f"SELL {position} shares at ${actual_price:.2f} on {date} ({reason}). P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        return capital, 0
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if self.equity_curve.empty:
            return {}
        
        returns = self.equity_curve['returns'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (self.equity_curve['total'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate annualized metrics
        days = len(returns)
        years = days / 252  # Trading days
        
        cagr = ((self.equity_curve['total'].iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = self.equity_curve['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        if self.trades:
            winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            total_trades = sum(1 for t in self.trades if 'pnl' in t)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Average win/loss
            wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        metrics = {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'volatility_pct': returns.std() * np.sqrt(252) * 100,
            'total_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.equity_curve['total'].iloc[-1]
        }
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate a detailed performance report"""
        metrics = self.get_performance_metrics()
        
        if not metrics:
            return "No backtest results available"
        
        report = f"""
{'='*60}
BACKTEST PERFORMANCE REPORT
{'='*60}

Initial Capital:        ${self.initial_capital:,.2f}
Final Capital:          ${metrics['final_capital']:,.2f}

RETURNS
-------
Total Return:           {metrics['total_return_pct']:.2f}%
CAGR:                   {metrics['cagr_pct']:.2f}%
Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%

RISK METRICS
------------
Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}
Sortino Ratio:          {metrics['sortino_ratio']:.2f}
Volatility (Annual):    {metrics['volatility_pct']:.2f}%

TRADING STATISTICS
------------------
Total Trades:           {metrics['total_trades']}
Win Rate:               {metrics['win_rate_pct']:.2f}%
Average Win:            ${metrics['avg_win']:,.2f}
Average Loss:           ${metrics['avg_loss']:,.2f}
Profit Factor:          {metrics['profit_factor']:.2f}

{'='*60}
"""
        return report
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)


if __name__ == "__main__":
    # Test backtester
    from data_handler import DataHandler
    from strategies import MACDStrategy
    
    # Fetch data
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2022-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        
        # Generate signals
        strategy = MACDStrategy()
        df = strategy.generate_signals(df)
        
        # Run backtest
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(df)
        
        # Print report
        print(backtester.generate_report())
        
        # Show trades
        trades_df = backtester.get_trades_df()
        print("\nTrades:")
        print(trades_df)
