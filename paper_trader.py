"""
Paper Trading System - Simulated live trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
from data_handler import DataHandler
from strategies import BaseStrategy
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position"""
    
    def __init__(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        entry_date: datetime,
        stop_loss: float = None,
        take_profit: float = None
    ):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current P&L"""
        return (current_price - self.entry_price) * self.shares
    
    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate current P&L percentage"""
        return ((current_price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class PaperTradingAccount:
    """Paper trading account management"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        max_position_size: float = 0.20
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions = []
        self.trade_history = []
        
        self.data_handler = DataHandler()
    
    def get_total_value(self) -> float:
        """Calculate total account value"""
        positions_value = sum(
            pos.shares * self.data_handler.get_live_price(symbol)
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def can_buy(self, symbol: str, shares: int, price: float) -> bool:
        """Check if we can buy"""
        cost = shares * price * (1 + self.commission)
        
        # Check if we have enough cash
        if cost > self.cash:
            return False
        
        # Check position size limit
        total_value = self.get_total_value()
        position_value = shares * price
        
        if position_value > total_value * self.max_position_size:
            return False
        
        return True
    
    def buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> bool:
        """
        Execute buy order
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            price: Current price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        
        Returns:
            True if successful
        """
        if not self.can_buy(symbol, shares, price):
            logger.warning(f"Cannot buy {symbol}: insufficient funds or position limit")
            return False
        
        # Calculate costs
        cost = shares * price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        # Update cash
        self.cash -= total_cost
        
        # Create position
        stop_loss = price * (1 - stop_loss_pct) if stop_loss_pct else None
        take_profit = price * (1 + take_profit_pct) if take_profit_pct else None
        
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': commission_cost,
            'total_cost': total_cost,
            'cash_remaining': self.cash
        }
        self.trade_history.append(trade)
        
        logger.info(f"BUY {shares} {symbol} @ ${price:.2f} (Cost: ${total_cost:.2f})")
        
        return True
    
    def sell(self, symbol: str, price: float, reason: str = 'signal') -> bool:
        """
        Execute sell order
        
        Args:
            symbol: Stock symbol
            price: Current price
            reason: Reason for selling
        
        Returns:
            True if successful
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol}: no position")
            return False
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        proceeds = position.shares * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        # Calculate P&L
        pnl = position.calculate_pnl(price) - commission_cost
        pnl_pct = position.calculate_pnl_pct(price)
        
        # Update cash
        self.cash += net_proceeds
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'SELL',
            'symbol': symbol,
            'shares': position.shares,
            'price': price,
            'commission': commission_cost,
            'proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_days': (datetime.now() - position.entry_date).days,
            'cash_remaining': self.cash
        }
        self.trade_history.append(trade)
        
        # Move to closed positions
        self.closed_positions.append({
            'position': position.to_dict(),
            'exit_price': price,
            'exit_date': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"SELL {position.shares} {symbol} @ ${price:.2f} ({reason}). P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        return True
    
    def check_stop_loss_take_profit(self):
        """Check all positions for stop loss or take profit"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_price = self.data_handler.get_live_price(symbol)
            
            if current_price is None:
                continue
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                positions_to_close.append((symbol, current_price, 'stop_loss'))
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                positions_to_close.append((symbol, current_price, 'take_profit'))
        
        # Close positions
        for symbol, price, reason in positions_to_close:
            self.sell(symbol, price, reason)
    
    def get_account_summary(self) -> Dict:
        """Get account summary"""
        total_value = self.get_total_value()
        
        positions_summary = []
        for symbol, pos in self.positions.items():
            current_price = self.data_handler.get_live_price(symbol)
            if current_price:
                positions_summary.append({
                    'symbol': symbol,
                    'shares': pos.shares,
                    'entry_price': pos.entry_price,
                    'current_price': current_price,
                    'pnl': pos.calculate_pnl(current_price),
                    'pnl_pct': pos.calculate_pnl_pct(current_price),
                    'value': pos.shares * current_price
                })
        
        return {
            'timestamp': datetime.now(),
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_value': total_value,
            'total_return': (total_value / self.initial_capital - 1) * 100,
            'positions': positions_summary,
            'num_positions': len(self.positions),
            'total_trades': len(self.trade_history)
        }
    
    def print_summary(self):
        """Print account summary"""
        summary = self.get_account_summary()
        
        print(f"\n{'='*60}")
        print("PAPER TRADING ACCOUNT SUMMARY")
        print(f"{'='*60}")
        print(f"Cash:              ${summary['cash']:,.2f}")
        print(f"Positions Value:   ${summary['positions_value']:,.2f}")
        print(f"Total Value:       ${summary['total_value']:,.2f}")
        print(f"Total Return:      {summary['total_return']:.2f}%")
        print(f"Open Positions:    {summary['num_positions']}")
        print(f"Total Trades:      {summary['total_trades']}")
        
        if summary['positions']:
            print(f"\n{'='*60}")
            print("OPEN POSITIONS")
            print(f"{'='*60}")
            for pos in summary['positions']:
                print(f"{pos['symbol']}: {pos['shares']} shares @ ${pos['current_price']:.2f}")
                print(f"  P&L: ${pos['pnl']:.2f} ({pos['pnl_pct']:.2f}%)")
        
        print(f"{'='*60}\n")
    
    def save_state(self, filename: str = 'paper_trading_state.json'):
        """Save account state to file"""
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'closed_positions': self.closed_positions,
            'trade_history': [
                {**trade, 'timestamp': trade['timestamp'].isoformat()} 
                for trade in self.trade_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Account state saved to {filename}")


class PaperTrader:
    """Paper trading system with strategy execution"""
    
    def __init__(
        self,
        account: PaperTradingAccount,
        strategy: BaseStrategy,
        symbols: List[str],
        check_interval: int = 60
    ):
        self.account = account
        self.strategy = strategy
        self.symbols = symbols
        self.check_interval = check_interval
        
        self.data_handler = DataHandler()
        self.running = False
    
    def start(self, duration_minutes: Optional[int] = None):
        """
        Start paper trading
        
        Args:
            duration_minutes: How long to run (None = indefinite)
        """
        self.running = True
        start_time = datetime.now()
        
        logger.info(f"Starting paper trading for {self.symbols}")
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"Check interval: {self.check_interval}s")
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration} - {datetime.now()}")
                logger.info(f"{'='*60}")
                
                # Check stop loss / take profit
                self.account.check_stop_loss_take_profit()
                
                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Print account summary
                self.account.print_summary()
                
                # Check duration
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        logger.info(f"Duration reached ({duration_minutes} minutes)")
                        break
                
                # Wait before next check
                logger.info(f"Waiting {self.check_interval} seconds...")
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("\nPaper trading stopped by user")
        
        finally:
            self.running = False
            self.account.save_state()
            logger.info("Paper trading session ended")
    
    def _process_symbol(self, symbol: str):
        """Process trading logic for a symbol"""
        try:
            # Get recent data
            df = self.data_handler.get_latest_data(symbol, period='3mo')
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return
            
            # Generate signals
            df = self.strategy.generate_signals(df)
            
            # Get latest signal
            latest_signal = df['signal'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Check if we have a position
            position = self.account.get_position(symbol)
            
            if latest_signal == 1 and position is None:
                # Buy signal
                total_value = self.account.get_total_value()
                max_position = total_value * self.account.max_position_size
                shares = int(max_position / current_price)
                
                if shares > 0:
                    self.account.buy(symbol, shares, current_price)
            
            elif latest_signal == -1 and position is not None:
                # Sell signal
                self.account.sell(symbol, current_price, 'signal')
            
            else:
                logger.info(f"{symbol}: No action (signal={latest_signal}, position={'exists' if position else 'none'})")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def stop(self):
        """Stop paper trading"""
        self.running = False


if __name__ == "__main__":
    from strategies import MACDStrategy
    
    # Create paper trading account
    account = PaperTradingAccount(initial_capital=100000)
    
    # Create strategy
    strategy = MACDStrategy()
    
    # Create paper trader
    trader = PaperTrader(
        account=account,
        strategy=strategy,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        check_interval=300  # 5 minutes
    )
    
    # Start paper trading for 1 hour
    trader.start(duration_minutes=60)
