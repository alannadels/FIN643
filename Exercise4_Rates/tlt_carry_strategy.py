"""
TLT Carry Strategy with Multi-Signal Risk Management
=====================================================

Strategy Overview:
- Base position: Long TLT (iShares 20+ Year Treasury Bond ETF)
- Objective: Capture carry/roll-down yield from long-duration Treasuries
- Risk Management: Exit to cash when multiple technical/volatility signals indicate risk-off

Exit Signals:
1. Moving Average Crossover: Fast MA crosses below Slow MA (trend deterioration)
2. RSI Overbought: RSI > 75 indicates potential pullback
3. VIX Spike: VIX > 25 or VIX increases > 20% from 10-day average
4. Volatility Regime: Realized vol exceeds 75th percentile (high volatility environment)
5. Drawdown Control: Exit if position drawdown exceeds 5%

Author: [Your Name]
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ======================== STRATEGY PARAMETERS ========================
LOOKBACK_YEARS = 10              # Historical data to fetch
NOTIONAL = 1_000_000             # Portfolio size
TRANSACTION_COST = 0.0005        # 5 bps per trade (conservative for ETFs)
LEVERAGE = 1.0                   # No leverage (leverage doesn't improve Sharpe)

# Moving Average Parameters
FAST_MA = 20                     # Fast MA period (days)
SLOW_MA = 50                     # Slow MA period (days)

# RSI Parameters
RSI_PERIOD = 14                  # Standard RSI period
RSI_OVERBOUGHT = 75              # Exit when RSI exceeds this

# VIX Parameters
VIX_THRESHOLD = 25               # Absolute VIX threshold
VIX_SPIKE_PCT = 0.20             # 20% increase from 10-day average

# Volatility Regime Parameters
VOL_WINDOW = 20                  # Realized volatility calculation window
VOL_REGIME_WINDOW = 252          # Rolling window to determine vol percentile
VOL_REGIME_THRESHOLD = 0.75      # Exit when vol > 75th percentile

# Drawdown Control
MAX_DRAWDOWN_PCT = 0.05          # Exit if drawdown exceeds 5%

# Re-entry Parameters
REENTRY_COOLDOWN = 5             # Days to wait before re-entering after exit

ANNUALIZATION = 252              # Trading days per year


# ======================== DATA FETCHING ========================
def fetch_data(lookback_years=LOOKBACK_YEARS):
    """Fetch TLT and VIX data from Yahoo Finance"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_years * 365)

    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")

    # Fetch both TLT and VIX together
    tickers = yf.download(['TLT', '^VIX'], start=start_date, end=end_date, progress=False)

    # Extract Close prices - the MultiIndex structure is ('Price Type', 'Ticker')
    df = pd.DataFrame({
        'TLT': tickers['Close']['TLT'],
        'VIX': tickers['Close']['^VIX']
    })

    df = df.dropna()

    print(f"Data fetched: {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ======================== TECHNICAL INDICATORS ========================
def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_moving_averages(prices):
    """Calculate fast and slow moving averages"""
    fast_ma = prices.rolling(window=FAST_MA).mean()
    slow_ma = prices.rolling(window=SLOW_MA).mean()
    return fast_ma, slow_ma


def calculate_realized_volatility(returns, window=VOL_WINDOW):
    """Calculate annualized realized volatility"""
    return returns.rolling(window=window).std() * np.sqrt(ANNUALIZATION)


def calculate_vol_regime(realized_vol, window=VOL_REGIME_WINDOW):
    """Calculate volatility percentile rank (0-1)"""
    return realized_vol.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ======================== SIGNAL GENERATION ========================
def generate_signals(df):
    """Generate entry/exit signals based on multiple factors"""
    signals = pd.DataFrame(index=df.index)

    # Price and returns
    signals['TLT'] = df['TLT']
    signals['Returns'] = df['TLT'].pct_change()

    # 1. Moving Average Signal
    signals['Fast_MA'], signals['Slow_MA'] = calculate_moving_averages(df['TLT'])
    signals['MA_Signal'] = (signals['Fast_MA'] > signals['Slow_MA']).astype(int)

    # 2. RSI Signal
    signals['RSI'] = calculate_rsi(df['TLT'])
    signals['RSI_Signal'] = (signals['RSI'] < RSI_OVERBOUGHT).astype(int)

    # 3. VIX Signals
    signals['VIX'] = df['VIX']
    signals['VIX_MA10'] = signals['VIX'].rolling(window=10).mean()
    signals['VIX_Absolute'] = (signals['VIX'] < VIX_THRESHOLD).astype(int)
    signals['VIX_Spike'] = ((signals['VIX'] / signals['VIX_MA10']) < (1 + VIX_SPIKE_PCT)).astype(int)
    signals['VIX_Signal'] = (signals['VIX_Absolute'] & signals['VIX_Spike']).astype(int)

    # 4. Volatility Regime Signal
    signals['Realized_Vol'] = calculate_realized_volatility(signals['Returns'])
    signals['Vol_Percentile'] = calculate_vol_regime(signals['Realized_Vol'])
    signals['Vol_Signal'] = (signals['Vol_Percentile'] < VOL_REGIME_THRESHOLD).astype(int)

    # Combined signal: All conditions must be true to hold position
    signals['Combined_Signal'] = (
        signals['MA_Signal'] &
        signals['RSI_Signal'] &
        signals['VIX_Signal'] &
        signals['Vol_Signal']
    ).astype(int)

    return signals.dropna()


# ======================== BACKTEST WITH DRAWDOWN CONTROL ========================
def run_backtest(signals):
    """Run backtest with position sizing and drawdown control"""
    bt = signals.copy()

    # Initialize position tracking
    bt['Position'] = 0
    bt['Trade_Signal'] = 0  # 1 = buy, -1 = sell, 0 = hold
    bt['Days_Since_Exit'] = 0
    bt['Peak_Value'] = NOTIONAL
    bt['Portfolio_Value'] = NOTIONAL

    position = 0
    days_since_exit = 0
    portfolio_value = NOTIONAL
    peak_value = NOTIONAL
    shares = 0

    for i in range(len(bt)):
        if i == 0:
            continue

        current_price = bt['TLT'].iloc[i]
        signal = bt['Combined_Signal'].iloc[i]

        # Update portfolio value
        if position == 1:
            portfolio_value = shares * current_price

        # Update peak value for drawdown calculation
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        # Calculate drawdown
        drawdown = (peak_value - portfolio_value) / peak_value

        # Increment cooldown counter
        if days_since_exit > 0:
            days_since_exit += 1
            if days_since_exit > REENTRY_COOLDOWN:
                days_since_exit = 0

        # Position logic
        if position == 0:  # Currently flat
            # Enter long if signal is positive and cooldown expired
            if signal == 1 and days_since_exit == 0:
                shares = portfolio_value / current_price
                position = 1
                bt.loc[bt.index[i], 'Trade_Signal'] = 1
                # Apply transaction cost
                portfolio_value *= (1 - TRANSACTION_COST)

        else:  # Currently long
            # Exit if signal turns negative OR drawdown exceeds threshold
            if signal == 0 or drawdown > MAX_DRAWDOWN_PCT:
                portfolio_value = shares * current_price
                # Apply transaction cost
                portfolio_value *= (1 - TRANSACTION_COST)
                shares = 0
                position = 0
                days_since_exit = 1
                bt.loc[bt.index[i], 'Trade_Signal'] = -1
                # Reset peak after exit
                peak_value = portfolio_value

        bt.loc[bt.index[i], 'Position'] = position
        bt.loc[bt.index[i], 'Days_Since_Exit'] = days_since_exit
        bt.loc[bt.index[i], 'Portfolio_Value'] = portfolio_value
        bt.loc[bt.index[i], 'Peak_Value'] = peak_value

    # Calculate daily returns
    bt['Portfolio_Returns_Unlevered'] = bt['Portfolio_Value'].pct_change().fillna(0)

    # Apply leverage to returns when in position
    bt['Portfolio_Returns'] = bt['Portfolio_Returns_Unlevered'] * np.where(bt['Position'] == 1, LEVERAGE, 1.0)

    # Recalculate portfolio value with leveraged returns
    bt['Portfolio_Value_Levered'] = NOTIONAL * (1 + bt['Portfolio_Returns']).cumprod()

    # Calculate buy-and-hold benchmark
    bt['BuyHold_Value'] = NOTIONAL * (bt['TLT'] / bt['TLT'].iloc[0])
    bt['BuyHold_Returns'] = bt['BuyHold_Value'].pct_change().fillna(0)

    return bt


# ======================== PERFORMANCE METRICS ========================
def calculate_performance_metrics(bt):
    """Calculate comprehensive performance statistics"""

    # Strategy metrics (use leveraged portfolio value)
    final_value = bt['Portfolio_Value_Levered'].iloc[-1]
    strategy_total_return = (final_value / NOTIONAL) - 1
    strategy_cagr = (final_value / NOTIONAL) ** (ANNUALIZATION / len(bt)) - 1

    strategy_daily_returns = bt['Portfolio_Returns']
    strategy_annual_return = strategy_daily_returns.mean() * ANNUALIZATION
    strategy_annual_vol = strategy_daily_returns.std() * np.sqrt(ANNUALIZATION)
    strategy_sharpe = strategy_annual_return / strategy_annual_vol if strategy_annual_vol > 0 else 0

    # Calculate maximum drawdown
    cumulative = (1 + strategy_daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    strategy_max_dd = drawdown.min()

    # Benchmark metrics
    benchmark_total_return = (bt['BuyHold_Value'].iloc[-1] / NOTIONAL) - 1
    benchmark_cagr = (bt['BuyHold_Value'].iloc[-1] / NOTIONAL) ** (ANNUALIZATION / len(bt)) - 1

    benchmark_daily_returns = bt['BuyHold_Returns']
    benchmark_annual_return = benchmark_daily_returns.mean() * ANNUALIZATION
    benchmark_annual_vol = benchmark_daily_returns.std() * np.sqrt(ANNUALIZATION)
    benchmark_sharpe = benchmark_annual_return / benchmark_annual_vol if benchmark_annual_vol > 0 else 0

    # Calculate benchmark max drawdown
    cumulative_bh = (1 + benchmark_daily_returns).cumprod()
    running_max_bh = cumulative_bh.expanding().max()
    drawdown_bh = (cumulative_bh - running_max_bh) / running_max_bh
    benchmark_max_dd = drawdown_bh.min()

    # Trading statistics
    trades = bt[bt['Trade_Signal'] != 0]
    num_trades = len(trades[trades['Trade_Signal'] == 1])  # Count buy signals
    days_in_market = (bt['Position'] == 1).sum()
    pct_time_in_market = days_in_market / len(bt)

    metrics = {
        'Strategy': {
            'Total Return': strategy_total_return,
            'CAGR': strategy_cagr,
            'Annual Return': strategy_annual_return,
            'Annual Volatility': strategy_annual_vol,
            'Sharpe Ratio': strategy_sharpe,
            'Max Drawdown': strategy_max_dd,
            'Final Value': final_value,
        },
        'Benchmark (Buy & Hold)': {
            'Total Return': benchmark_total_return,
            'CAGR': benchmark_cagr,
            'Annual Return': benchmark_annual_return,
            'Annual Volatility': benchmark_annual_vol,
            'Sharpe Ratio': benchmark_sharpe,
            'Max Drawdown': benchmark_max_dd,
            'Final Value': bt['BuyHold_Value'].iloc[-1],
        },
        'Trading Stats': {
            'Number of Trades': num_trades,
            'Days in Market': days_in_market,
            '% Time in Market': pct_time_in_market,
        }
    }

    return metrics


def print_performance_summary(metrics):
    """Print formatted performance summary"""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY: TLT CARRY STRATEGY WITH RISK MANAGEMENT")
    print("="*70)

    print("\n--- STRATEGY PERFORMANCE ---")
    for key, value in metrics['Strategy'].items():
        if key == 'Final Value':
            print(f"{key:.<30} ${value:,.2f}")
        elif 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
            print(f"{key:.<30} {value:.2%}")
        else:
            print(f"{key:.<30} {value:.4f}")

    print("\n--- BENCHMARK PERFORMANCE (Buy & Hold TLT) ---")
    for key, value in metrics['Benchmark (Buy & Hold)'].items():
        if key == 'Final Value':
            print(f"{key:.<30} ${value:,.2f}")
        elif 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
            print(f"{key:.<30} {value:.2%}")
        else:
            print(f"{key:.<30} {value:.4f}")

    print("\n--- TRADING STATISTICS ---")
    for key, value in metrics['Trading Stats'].items():
        if '% Time' in key:
            print(f"{key:.<30} {value:.2%}")
        else:
            print(f"{key:.<30} {value}")

    print("\n" + "="*70)
    print(f"*** KEY METRIC: SHARPE RATIO = {metrics['Strategy']['Sharpe Ratio']:.4f} ***")
    print("="*70 + "\n")


# ======================== VISUALIZATION ========================
def plot_results(bt, metrics):
    """Create comprehensive visualization of strategy performance"""

    fig, axes = plt.subplots(5, 1, figsize=(14, 16))

    # 1. Portfolio Value Comparison
    ax1 = axes[0]
    ax1.plot(bt.index, bt['Portfolio_Value_Levered'], label=f'Strategy ({LEVERAGE:.1f}x Leverage)', linewidth=2, color='darkblue')
    ax1.plot(bt.index, bt['BuyHold_Value'], label='Buy & Hold', linewidth=2,
             color='gray', alpha=0.7, linestyle='--')
    ax1.set_title('Portfolio Value: Leveraged Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=NOTIONAL, color='red', linestyle=':', alpha=0.5, label='Initial Capital')

    # 2. TLT Price with Position Overlay
    ax2 = axes[1]
    ax2.plot(bt.index, bt['TLT'], label='TLT Price', color='black', linewidth=1.5)
    ax2.plot(bt.index, bt['Fast_MA'], label=f'{FAST_MA}d MA', color='blue', linewidth=1, alpha=0.7)
    ax2.plot(bt.index, bt['Slow_MA'], label=f'{SLOW_MA}d MA', color='red', linewidth=1, alpha=0.7)

    # Highlight position periods
    in_position = bt['Position'] == 1
    ax2.fill_between(bt.index, bt['TLT'].min(), bt['TLT'].max(),
                      where=in_position, alpha=0.1, color='green', label='Long Position')

    ax2.set_title('TLT Price with Moving Averages & Position Periods', fontsize=12, fontweight='bold')
    ax2.set_ylabel('TLT Price ($)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. RSI
    ax3 = axes[2]
    ax3.plot(bt.index, bt['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax3.axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', label=f'Overbought ({RSI_OVERBOUGHT})')
    ax3.axhline(y=30, color='green', linestyle='--', label='Oversold (30)', alpha=0.5)
    ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RSI')
    ax3.set_ylim([0, 100])
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. VIX and Volatility
    ax4 = axes[3]
    ax4_twin = ax4.twinx()

    ax4.plot(bt.index, bt['VIX'], label='VIX', color='orange', linewidth=1.5)
    ax4.axhline(y=VIX_THRESHOLD, color='red', linestyle='--', label=f'VIX Threshold ({VIX_THRESHOLD})')
    ax4.set_ylabel('VIX Level', color='orange')
    ax4.tick_params(axis='y', labelcolor='orange')

    ax4_twin.plot(bt.index, bt['Realized_Vol']*100, label='Realized Vol (%)',
                  color='brown', linewidth=1.5, alpha=0.7)
    ax4_twin.set_ylabel('Realized Volatility (%)', color='brown')
    ax4_twin.tick_params(axis='y', labelcolor='brown')

    ax4.set_title('VIX and Realized Volatility', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # 5. Drawdown Comparison
    ax5 = axes[4]

    # Calculate drawdowns
    strategy_cumulative = (1 + bt['Portfolio_Returns']).cumprod()
    strategy_running_max = strategy_cumulative.expanding().max()
    strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max * 100

    benchmark_cumulative = (1 + bt['BuyHold_Returns']).cumprod()
    benchmark_running_max = benchmark_cumulative.expanding().max()
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max * 100

    ax5.fill_between(bt.index, strategy_drawdown, 0, label='Strategy DD',
                     color='darkblue', alpha=0.5)
    ax5.plot(bt.index, benchmark_drawdown, label='Buy & Hold DD',
             color='red', linewidth=1.5, alpha=0.7)
    ax5.axhline(y=-MAX_DRAWDOWN_PCT*100, color='red', linestyle='--',
                label=f'DD Threshold (-{MAX_DRAWDOWN_PCT*100:.0f}%)')
    ax5.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Drawdown (%)')
    ax5.set_xlabel('Date')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/alannadels/Desktop/Finance/exercise4_rates/tlt_strategy_performance.png',
                dpi=300, bbox_inches='tight')
    print("Chart saved: tlt_strategy_performance.png")
    plt.show()


# ======================== MAIN EXECUTION ========================
def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("TLT CARRY STRATEGY WITH MULTI-SIGNAL RISK MANAGEMENT")
    print("="*70 + "\n")

    # Fetch data
    df = fetch_data()

    # Generate signals
    print("\nGenerating trading signals...")
    signals = generate_signals(df)

    # Run backtest
    print("Running backtest with drawdown control...")
    bt = run_backtest(signals)

    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(bt)

    # Print results
    print_performance_summary(metrics)

    # Plot results
    print("Generating visualizations...")
    plot_results(bt, metrics)

    # Export results
    results_file = '/Users/alannadels/Desktop/Finance/exercise4_rates/backtest_results.csv'
    bt.to_csv(results_file)
    print(f"\nBacktest results exported: {results_file}")

    return bt, metrics


if __name__ == "__main__":
    bt, metrics = main()
