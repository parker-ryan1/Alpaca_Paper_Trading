"""
Live Paper Trading with Alpaca API
Sign up for free at https://alpaca.markets/
"""
import alpaca_trade_api as tradeapi
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config
from data_handler import DataHandler
from strategies import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpacaPaperTrader:
    """
    Live paper trading using Alpaca's paper trading API
    
    Features:
    - Real market data
    - Realistic order execution
    - Paper trading account (no real money)
    - Full broker integration
    """
    
    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        base_url: str = None
    ):
        """
        Initialize Alpaca connection
        
        Args:
            api_key: Alpaca API key (or set in .env)
            secret_key: Alpaca secret key (or set in .env)
            base_url: Paper trading URL
        """
        self.api_key = api_key or config.ALPACA_API_KEY
        self.secret_key = secret_key or config.ALPACA_SECRET_KEY
        self.base_url = base_url or config.ALPACA_BASE_URL
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. "
                "Sign up at https://alpaca.markets/ and set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )
        
        self.data_handler = DataHandler()
        self.positions = {}
        self.running = False
        
        logger.info("Connected to Alpaca Paper Trading")
        
        # Verify connection
        try:
            account = self.api.get_account()
            logger.info(f"Account Status: {account.status}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'status': account.status,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol"""
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        except:
            return None
    
    def get_buying_power(self) -> float:
        """Get available buying power"""
        account = self.api.get_account()
        return float(account.buying_power)
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_market_hours(self) -> Dict:
        """Get market hours information"""
        try:
            clock = self.api.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting market hours: {e}")
            return {}
    
    def buy_market(
        self,
        symbol: str,
        qty: int,
        stop_loss_pct: float = None,
        take_profit_pct: float = None
    ) -> bool:
        """
        Place market buy order
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
        
        Returns:
            True if successful
        """
        try:
            # Submit market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"BUY order submitted: {qty} shares of {symbol} (Order ID: {order.id})")
            
            # Wait for order to fill
            time.sleep(2)
            order = self.api.get_order(order.id)
            
            if order.status == 'filled':
                filled_price = float(order.filled_avg_price)
                logger.info(f"Order FILLED: {qty} {symbol} @ ${filled_price:.2f}")
                
                # Place stop loss and take profit orders if specified
                if stop_loss_pct or take_profit_pct:
                    self._place_bracket_orders(
                        symbol, qty, filled_price, stop_loss_pct, take_profit_pct
                    )
                
                return True
            else:
                logger.warning(f"Order status: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return False
    
    def sell_market(self, symbol: str, qty: int = None) -> bool:
        """
        Place market sell order
        
        Args:
            symbol: Stock symbol
            qty: Number of shares (None = sell all)
        
        Returns:
            True if successful
        """
        try:
            # If qty not specified, sell entire position
            if qty is None:
                position = self.get_position(symbol)
                if position:
                    qty = position['qty']
                else:
                    logger.warning(f"No position in {symbol}")
                    return False
            
            # Submit market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"SELL order submitted: {qty} shares of {symbol} (Order ID: {order.id})")
            
            # Wait for order to fill
            time.sleep(2)
            order = self.api.get_order(order.id)
            
            if order.status == 'filled':
                filled_price = float(order.filled_avg_price)
                logger.info(f"Order FILLED: {qty} {symbol} @ ${filled_price:.2f}")
                
                # Cancel any existing stop/take profit orders
                self._cancel_orders_for_symbol(symbol)
                
                return True
            else:
                logger.warning(f"Order status: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")
            return False
    
    def _place_bracket_orders(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ):
        """Place stop loss and take profit orders"""
        try:
            # Calculate prices
            stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
            limit_price = entry_price * (1 + take_profit_pct) if take_profit_pct else None
            
            # Place stop loss
            if stop_price:
                stop_order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='stop',
                    time_in_force='gtc',
                    stop_price=round(stop_price, 2)
                )
                logger.info(f"Stop loss placed at ${stop_price:.2f}")
            
            # Place take profit
            if limit_price:
                limit_order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='limit',
                    time_in_force='gtc',
                    limit_price=round(limit_price, 2)
                )
                logger.info(f"Take profit placed at ${limit_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error placing bracket orders: {e}")
    
    def _cancel_orders_for_symbol(self, symbol: str):
        """Cancel all open orders for a symbol"""
        try:
            orders = self.api.list_orders(status='open')
            for order in orders:
                if order.symbol == symbol:
                    self.api.cancel_order(order.id)
                    logger.info(f"Cancelled order {order.id} for {symbol}")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            logger.info("Cancelled all open orders")
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            self.api.close_all_positions()
            logger.info("Closed all positions")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def print_summary(self):
        """Print account and positions summary"""
        account = self.get_account()
        positions = self.get_positions()
        
        print(f"\n{'='*70}")
        print("ALPACA PAPER TRADING ACCOUNT")
        print(f"{'='*70}")
        print(f"Status:           {account.get('status', 'N/A')}")
        print(f"Portfolio Value:  ${account.get('portfolio_value', 0):,.2f}")
        print(f"Cash:             ${account.get('cash', 0):,.2f}")
        print(f"Buying Power:     ${account.get('buying_power', 0):,.2f}")
        
        if positions:
            print(f"\n{'='*70}")
            print("OPEN POSITIONS")
            print(f"{'='*70}")
            for pos in positions:
                print(f"\n{pos['symbol']}: {pos['qty']} shares")
                print(f"  Entry Price:  ${pos['avg_entry_price']:.2f}")
                print(f"  Current:      ${pos['current_price']:.2f}")
                print(f"  Market Value: ${pos['market_value']:,.2f}")
                print(f"  P&L:          ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2%})")
        else:
            print("\nNo open positions")
        
        print(f"{'='*70}\n")
    
    def start_trading(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        check_interval: int = 60,
        position_size_pct: float = 0.2,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Start automated trading
        
        Args:
            strategy: Trading strategy to use
            symbols: List of symbols to trade
            check_interval: How often to check signals (seconds)
            position_size_pct: Max position size as % of portfolio
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.running = True
        iteration = 0
        
        logger.info(f"\n{'='*70}")
        logger.info("STARTING ALPACA PAPER TRADING")
        logger.info(f"{'='*70}")
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Check Interval: {check_interval}s")
        logger.info(f"{'='*70}\n")
        
        try:
            while self.running:
                iteration += 1
                
                # Check if market is open
                if not self.is_market_open():
                    market_hours = self.get_market_hours()
                    logger.info(f"Market is closed. Next open: {market_hours.get('next_open')}")
                    time.sleep(300)  # Check again in 5 minutes
                    continue
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Iteration {iteration} - {datetime.now()}")
                logger.info(f"{'='*70}")
                
                # Process each symbol
                for symbol in symbols:
                    self._process_symbol_alpaca(
                        symbol,
                        strategy,
                        position_size_pct,
                        stop_loss_pct,
                        take_profit_pct
                    )
                
                # Print summary
                self.print_summary()
                
                # Wait
                logger.info(f"Waiting {check_interval} seconds...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n\nTrading stopped by user")
        finally:
            self.running = False
            logger.info("Trading session ended")
    
    def _process_symbol_alpaca(
        self,
        symbol: str,
        strategy: BaseStrategy,
        position_size_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ):
        """Process trading logic for a symbol"""
        try:
            # Get recent data
            df = self.data_handler.get_latest_data(symbol, period='3mo')
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return
            
            # Generate signals
            df = strategy.generate_signals(df)
            
            # Get latest signal
            latest_signal = df['signal'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Check current position
            position = self.get_position(symbol)
            
            if latest_signal == 1 and position is None:
                # BUY signal and no position
                account = self.get_account()
                portfolio_value = account['portfolio_value']
                
                # Calculate position size
                max_position_value = portfolio_value * position_size_pct
                qty = int(max_position_value / current_price)
                
                if qty > 0:
                    logger.info(f"📈 BUY SIGNAL for {symbol} @ ${current_price:.2f}")
                    self.buy_market(symbol, qty, stop_loss_pct, take_profit_pct)
                else:
                    logger.warning(f"Cannot buy {symbol}: insufficient buying power")
            
            elif latest_signal == -1 and position is not None:
                # SELL signal and have position
                logger.info(f"📉 SELL SIGNAL for {symbol} @ ${current_price:.2f}")
                self.sell_market(symbol)
            
            else:
                status = "Position exists" if position else "No position"
                logger.info(f"{symbol}: No action (Signal: {latest_signal}, {status})")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False


if __name__ == "__main__":
    """
    Test Alpaca connection
    
    Before running:
    1. Sign up at https://alpaca.markets/
    2. Get your paper trading API keys
    3. Add them to .env file:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
    """
    try:
        # Initialize trader
        trader = AlpacaPaperTrader()
        
        # Print account info
        trader.print_summary()
        
        # Check market status
        hours = trader.get_market_hours()
        print(f"\nMarket Open: {hours['is_open']}")
        
        # Example: Start trading (uncomment to run)
        # from strategies import MACDStrategy
        # strategy = MACDStrategy()
        # trader.start_trading(
        #     strategy=strategy,
        #     symbols=['AAPL', 'MSFT'],
        #     check_interval=300  # 5 minutes
        # )
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you've set up your Alpaca credentials in .env file")
