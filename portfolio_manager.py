"""
Portfolio Management System
Tracks positions, trades, and performance
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    current_price: float
    
    def get_market_value(self) -> float:
        """Get current market value"""
        return self.shares * self.current_price
    
    def get_cost_basis(self) -> float:
        """Get cost basis"""
        return self.shares * self.entry_price
    
    def get_pnl(self) -> float:
        """Get profit/loss in dollars"""
        return self.get_market_value() - self.get_cost_basis()
    
    def get_pnl_pct(self) -> float:
        """Get profit/loss as percentage"""
        return (self.get_pnl() / self.get_cost_basis()) * 100 if self.get_cost_basis() > 0 else 0


class PortfolioManager:
    """
    Portfolio Management System
    
    Features:
    - Track positions and cash
    - Record trade history
    - Calculate performance metrics
    - Risk management
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_curve = pd.DataFrame()
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total profit/loss"""
        return self.get_total_value() - self.initial_capital
    
    def get_total_pnl_pct(self) -> float:
        """Get total profit/loss percentage"""
        return (self.get_total_pnl() / self.initial_capital) * 100
    
    def open_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: Optional[datetime] = None
    ) -> bool:
        """Open a new position"""
        if date is None:
            date = datetime.now()
        
        cost = shares * price
        
        if cost > self.cash:
            print(f"Insufficient cash: ${self.cash:.2f} needed ${cost:.2f}")
            return False
        
        if symbol in self.positions:
            print(f"Position already exists for {symbol}")
            return False
        
        # Deduct cash
        self.cash -= cost
        
        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price
        )
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'cost': cost
        })
        
        return True
    
    def close_position(
        self,
        symbol: str,
        price: float,
        date: Optional[datetime] = None
    ) -> bool:
        """Close an existing position"""
        if date is None:
            date = datetime.now()
        
        if symbol not in self.positions:
            print(f"No position exists for {symbol}")
            return False
        
        position = self.positions[symbol]
        proceeds = position.shares * price
        
        # Add cash
        self.cash += proceeds
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': position.shares,
            'price': price,
            'proceeds': proceeds,
            'pnl': position.get_pnl()
        })
        
        # Remove position
        del self.positions[symbol]
        
        return True
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation by symbol"""
        total_value = self.get_total_value()
        
        allocation = {
            'CASH': (self.cash / total_value) * 100
        }
        
        for symbol, position in self.positions.items():
            allocation[symbol] = (position.get_market_value() / total_value) * 100
        
        return allocation
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        
        winning_trades = [t for t in self.trade_history 
                         if t.get('action') == 'SELL' and t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history 
                        if t.get('action') == 'SELL' and t.get('pnl', 0) < 0]
        
        total_trades = len([t for t in self.trade_history if t.get('action') == 'SELL'])
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': self.get_total_pnl_pct(),
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Shares': pos.shares,
                'Entry Price': pos.entry_price,
                'Current Price': pos.current_price,
                'Market Value': pos.get_market_value(),
                'Cost Basis': pos.get_cost_basis(),
                'P&L': pos.get_pnl(),
                'P&L %': pos.get_pnl_pct(),
                'Entry Date': pos.entry_date
            })
        
        return pd.DataFrame(data)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate portfolio Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        returns = self.equity_curve['total'].pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.equity_curve) == 0:
            return 0.0
        
        cumulative = self.equity_curve['total']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min() * 100
    
    def export_trades(self, filename: str = 'trades.csv'):
        """Export trade history to CSV"""
        df = pd.DataFrame(self.trade_history)
        df.to_csv(filename, index=False)
        print(f"Trade history exported to {filename}")
    
    def __str__(self) -> str:
        """String representation"""
        summary = self.get_performance_summary()
        
        return f"""
Portfolio Summary
=================
Total Value:     ${summary['total_value']:,.2f}
Cash:            ${summary['cash']:,.2f}
P&L:             ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:.2f}%)

Positions:       {summary['positions_count']}
Total Trades:    {summary['total_trades']}
Win Rate:        {summary['win_rate']:.2f}%
"""


if __name__ == "__main__":
    # Test portfolio manager
    pm = PortfolioManager(initial_capital=100000)
    
    # Open positions
    pm.open_position('AAPL', 100, 150.0)
    pm.open_position('MSFT', 50, 300.0)
    
    print(pm)
    print("\nPositions:")
    print(pm.get_position_summary())
    
    # Update prices
    pm.update_prices({'AAPL': 155.0, 'MSFT': 310.0})
    
    print("\nAfter price update:")
    print(pm)
    
    # Close position
    pm.close_position('AAPL', 155.0)
    
    print("\nAfter closing AAPL:")
    print(pm)
