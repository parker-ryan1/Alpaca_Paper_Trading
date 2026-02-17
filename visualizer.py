"""
Visualization tools for backtesting and trading results
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional

# Make optional imports
try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False


class Visualizer:
    """Visualization tools for trading analysis"""
    
    @staticmethod
    def plot_equity_curve(
        df: pd.DataFrame,
        title: str = "Equity Curve",
        benchmark: Optional[pd.Series] = None
    ):
        """Plot equity curve"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        ax1.plot(df.index, df['total'], label='Strategy', linewidth=2)
        
        if benchmark is not None:
            ax1.plot(df.index, benchmark, label='Benchmark', linewidth=2, alpha=0.7)
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = df['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        ax2.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_trades(df: pd.DataFrame, trades_df: pd.DataFrame, symbol: str = ''):
        """Plot price chart with buy/sell signals"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax.plot(df.index, df['close'], label='Price', linewidth=2, color='blue', alpha=0.7)
        
        # Plot moving averages if available
        if 'sma_20' in df.columns:
            ax.plot(df.index, df['sma_20'], label='SMA 20', linewidth=1, alpha=0.5)
        if 'sma_50' in df.columns:
            ax.plot(df.index, df['sma_50'], label='SMA 50', linewidth=1, alpha=0.5)
        
        # Plot buy signals
        buy_trades = trades_df[trades_df['type'] == 'BUY']
        if not buy_trades.empty:
            ax.scatter(
                buy_trades['date'],
                buy_trades['price'],
                marker='^',
                color='green',
                s=100,
                label='Buy',
                zorder=5
            )
        
        # Plot sell signals
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        if not sell_trades.empty:
            ax.scatter(
                sell_trades['date'],
                sell_trades['price'],
                marker='v',
                color='red',
                s=100,
                label='Sell',
                zorder=5
            )
        
        ax.set_title(f'Trading Signals - {symbol}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_indicators(df: pd.DataFrame, symbol: str = ''):
        """Plot price with technical indicators"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Price and MAs
        axes[0].plot(df.index, df['close'], label='Close', linewidth=2)
        if 'sma_20' in df.columns:
            axes[0].plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in df.columns:
            axes[0].plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
        
        # Bollinger Bands
        if 'bb_upper' in df.columns:
            axes[0].fill_between(
                df.index,
                df['bb_lower'],
                df['bb_upper'],
                alpha=0.2,
                label='BB'
            )
        
        axes[0].set_title(f'{symbol} - Price & Moving Averages', fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in df.columns:
            axes[1].plot(df.index, df['macd'], label='MACD', linewidth=2)
            axes[1].plot(df.index, df['macd_signal'], label='Signal', linewidth=2)
            axes[1].bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_title('MACD', fontweight='bold')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in df.columns:
            axes[2].plot(df.index, df['rsi'], label='RSI', linewidth=2, color='purple')
            axes[2].axhline(y=70, color='r', linestyle='--', label='Overbought')
            axes[2].axhline(y=30, color='g', linestyle='--', label='Oversold')
            axes[2].set_title('RSI', fontweight='bold')
            axes[2].set_ylim([0, 100])
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
        
        # Volume
        axes[3].bar(df.index, df['volume'], alpha=0.3, label='Volume')
        if 'volume_sma' in df.columns:
            axes[3].plot(df.index, df['volume_sma'], color='red', label='Volume MA', linewidth=2)
        axes[3].set_title('Volume', fontweight='bold')
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series):
        """Plot returns distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[0].axvline(returns.median(), color='g', linestyle='--', label=f'Median: {returns.median():.4f}')
        axes[0].set_title('Returns Distribution', fontweight='bold')
        axes[0].set_xlabel('Returns')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_monthly_returns(returns: pd.Series):
        """Plot monthly returns heatmap"""
        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame(monthly_returns, columns=['returns'])
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Returns')
        
        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not np.isnan(pivot.values[i, j]):
                    text = ax.text(j, i, f'{pivot.values[i, j]:.1%}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_candlestick(df: pd.DataFrame, symbol: str = '', days: int = 60):
        """Plot candlestick chart with volume"""
        if not HAS_MPLFINANCE:
            print("mplfinance not installed. Using standard plot instead.")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            df_plot = df.tail(days).copy()
            
            # Price plot
            ax1.plot(df_plot.index, df_plot['close'], label='Close', linewidth=2)
            if 'sma_20' in df_plot.columns:
                ax1.plot(df_plot.index, df_plot['sma_20'], label='SMA 20', alpha=0.7)
            if 'sma_50' in df_plot.columns:
                ax1.plot(df_plot.index, df_plot['sma_50'], label='SMA 50', alpha=0.7)
            ax1.set_title(f'{symbol} - Price Chart')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume plot
            ax2.bar(df_plot.index, df_plot['volume'], alpha=0.3)
            ax2.set_title('Volume')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            return
            
        # Get last N days
        df_plot = df.tail(days).copy()
        
        # Prepare data for mplfinance
        df_plot.index.name = 'Date'
        
        # Add moving averages if available
        addplot = []
        if 'sma_20' in df_plot.columns:
            addplot.append(mpf.make_addplot(df_plot['sma_20'], color='blue'))
        if 'sma_50' in df_plot.columns:
            addplot.append(mpf.make_addplot(df_plot['sma_50'], color='red'))
        
        # Plot
        kwargs = dict(
            type='candle',
            style='charles',
            title=f'{symbol} - Candlestick Chart',
            volume=True,
            figsize=(14, 8),
            addplot=addplot if addplot else None
        )
        
        mpf.plot(df_plot, **kwargs)
    
    @staticmethod
    def create_tearsheet(returns: pd.Series, benchmark: Optional[pd.Series] = None):
        """Create comprehensive performance tearsheet using quantstats"""
        if not HAS_QUANTSTATS:
            print("Quantstats not installed. Skipping tearsheet generation.")
            print("Install with: pip install quantstats")
            return
        qs.reports.html(returns, benchmark=benchmark, output='tearsheet.html')
        print("Tearsheet saved to tearsheet.html")


if __name__ == "__main__":
    # Test visualizations
    from data_handler import DataHandler
    from strategies import MACDStrategy
    from backtester import Backtester
    
    # Fetch data
    handler = DataHandler()
    data = handler.fetch_data(['AAPL'], '2022-01-01', '2024-01-01')
    
    if 'AAPL' in data:
        df = data['AAPL']
        
        # Generate signals
        strategy = MACDStrategy()
        df = strategy.generate_signals(df)
        
        # Run backtest
        backtester = Backtester()
        results = backtester.run(df)
        
        # Create visualizations
        viz = Visualizer()
        
        # Plot equity curve
        viz.plot_equity_curve(results, title="AAPL - MACD Strategy")
        
        # Plot trades
        trades_df = backtester.get_trades_df()
        viz.plot_trades(df, trades_df, symbol='AAPL')
        
        # Plot indicators
        viz.plot_indicators(df, symbol='AAPL')
