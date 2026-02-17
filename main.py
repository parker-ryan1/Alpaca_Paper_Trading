"""
Main entry point for the algorithmic trading system
"""
import argparse
import sys
from datetime import datetime
import pandas as pd

from data_handler import DataHandler
from strategies import get_strategy, STRATEGIES
from backtester import Backtester
from paper_trader import PaperTradingAccount, PaperTrader
from visualizer import Visualizer
import config


def run_backtest(args):
    """Run backtest on historical data"""
    print(f"\n{'='*60}")
    print("RUNNING BACKTEST")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"{'='*60}\n")
    
    # Initialize components
    handler = DataHandler()
    strategy = get_strategy(args.strategy)
    backtester = Backtester(
        initial_capital=args.capital,
        commission=config.COMMISSION
    )
    
    all_results = []
    
    # Run backtest for each symbol
    for symbol in args.symbols:
        print(f"\nProcessing {symbol}...")
        
        # Fetch data
        data = handler.fetch_data(
            [symbol],
            args.start_date,
            args.end_date
        )
        
        if symbol not in data:
            print(f"Skipping {symbol}: no data available")
            continue
        
        df = data[symbol]
        print(f"Loaded {len(df)} days of data")
        
        # Generate signals
        df = strategy.generate_signals(df)
        
        # Run backtest
        results = backtester.run(
            df,
            stop_loss_pct=config.STOP_LOSS_PCT,
            take_profit_pct=config.TAKE_PROFIT_PCT,
            position_size_pct=config.MAX_POSITION_SIZE
        )
        
        # Store results
        results['symbol'] = symbol
        all_results.append({
            'symbol': symbol,
            'results': results,
            'trades': backtester.get_trades_df()
        })
        
        # Print report
        print(backtester.generate_report())
        
        # Visualize if requested
        if args.plot:
            viz = Visualizer()
            viz.plot_equity_curve(results, title=f"{symbol} - {strategy.name}")
            viz.plot_trades(df, backtester.get_trades_df(), symbol=symbol)
            viz.plot_indicators(df, symbol=symbol)
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        symbol = result['symbol']
        df = result['results']
        trades = result['trades']
        
        final_value = df['total'].iloc[-1]
        total_return = (final_value / args.capital - 1) * 100
        num_trades = len(trades)
        
        print(f"\n{symbol}:")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Trades: {num_trades}")
    
    print(f"\n{'='*60}\n")
    
    # Save results if requested
    if args.save:
        for result in all_results:
            symbol = result['symbol']
            results_df = result['results']
            trades_df = result['trades']
            
            # Save to CSV
            results_df.to_csv(f'backtest_results_{symbol}.csv')
            trades_df.to_csv(f'backtest_trades_{symbol}.csv')
            
            print(f"Saved results for {symbol}")


def run_paper_trading(args):
    """Run paper trading session"""
    print(f"\n{'='*60}")
    print("STARTING PAPER TRADING")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Check Interval: {args.interval}s")
    print(f"{'='*60}\n")
    
    # Create account
    account = PaperTradingAccount(
        initial_capital=args.capital,
        commission=config.COMMISSION,
        max_position_size=config.MAX_POSITION_SIZE
    )
    
    # Create strategy
    strategy = get_strategy(args.strategy)
    
    # Create paper trader
    trader = PaperTrader(
        account=account,
        strategy=strategy,
        symbols=args.symbols,
        check_interval=args.interval
    )
    
    # Start trading
    try:
        trader.start(duration_minutes=args.duration)
    except KeyboardInterrupt:
        print("\nStopping paper trading...")
        trader.stop()
    
    # Final summary
    account.print_summary()


def list_strategies():
    """List available strategies"""
    print(f"\n{'='*60}")
    print("AVAILABLE STRATEGIES")
    print(f"{'='*60}\n")
    
    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class()
        print(f"  {name:<15} - {strategy.name}")
    
    print(f"\n{'='*60}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Algorithmic Trading System with Backtesting and Paper Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest MACD strategy on AAPL
  python main.py backtest --symbols AAPL --strategy macd --plot
  
  # Backtest multiple symbols
  python main.py backtest --symbols AAPL MSFT GOOGL --strategy rsi
  
  # Run paper trading for 2 hours
  python main.py paper --symbols AAPL MSFT --strategy macd --duration 120
  
  # List available strategies
  python main.py list-strategies
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Stock symbols to trade'
    )
    backtest_parser.add_argument(
        '--strategy',
        choices=list(STRATEGIES.keys()),
        default='macd',
        help='Trading strategy to use'
    )
    backtest_parser.add_argument(
        '--start-date',
        default=config.START_DATE,
        help='Start date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--end-date',
        default=config.END_DATE,
        help='End date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--capital',
        type=float,
        default=config.INITIAL_CAPITAL,
        help='Initial capital'
    )
    backtest_parser.add_argument(
        '--plot',
        action='store_true',
        help='Show plots'
    )
    backtest_parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to CSV'
    )
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Stock symbols to trade'
    )
    paper_parser.add_argument(
        '--strategy',
        choices=list(STRATEGIES.keys()),
        default='macd',
        help='Trading strategy to use'
    )
    paper_parser.add_argument(
        '--capital',
        type=float,
        default=config.INITIAL_CAPITAL,
        help='Initial capital'
    )
    paper_parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds'
    )
    paper_parser.add_argument(
        '--duration',
        type=int,
        help='Duration in minutes (default: indefinite)'
    )
    
    # List strategies command
    subparsers.add_parser('list-strategies', help='List available strategies')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'paper':
        run_paper_trading(args)
    elif args.command == 'list-strategies':
        list_strategies()


if __name__ == "__main__":
    main()
