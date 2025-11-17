"""
Yen Carry Trade - Strategy Implementation
==================================================

VIX Spike (20) Strategy:
- Borrow JPY at low interest rates
- Invest in tech portfolio: 50% GOOGL, 30% NVDA, 20% IBM
- Exit when VIX > 20 (high volatility/risk-off)
- Re-enter when VIX < 20 (low volatility/risk-on)

This strategy achieved:
- Sharpe Ratio: 3.85
- Total Return: 1,237.52%
- Max Drawdown: -16.10%

Author: Alan Nadelsticher
Date: November 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Portfolio allocation
ALLOCATION = {
    'GOOGL': 0.50,
    'NVDA': 0.30,
    'IBM': 0.20
}

# Time period
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# Strategy parameters
VIX_THRESHOLD = 20  # Exit when VIX exceeds this level
JPY_RATE = 0.001    # ~0.1% per year
USD_RATE = 0.035    # ~3.5% per year average

print("=" * 80)
print("YEN CARRY TRADE - WINNING STRATEGY (VIX SPIKE 20)")
print("=" * 80)
print(f"\nPortfolio: 50% GOOGL, 30% NVDA, 20% IBM")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Exit Rule: VIX > {VIX_THRESHOLD}")
print("\n" + "=" * 80)

# ============================================================================
# STEP 1: Download Data
# ============================================================================

print("\n[1/5] Downloading market data...")

# USD/JPY exchange rate
usdjpy = yf.download('USDJPY=X', start=START_DATE, end=END_DATE, progress=False)
if usdjpy.empty:
    usdjpy = yf.download('JPY=X', start=START_DATE, end=END_DATE, progress=False)

if 'Close' in usdjpy.columns:
    usdjpy['Rate'] = usdjpy['Close']
elif isinstance(usdjpy.columns, pd.MultiIndex):
    usdjpy['Rate'] = usdjpy['Close'].iloc[:, 0]
else:
    usdjpy['Rate'] = usdjpy.iloc[:, 0]

print(f"  [OK] USD/JPY: {len(usdjpy)} records")

# Download stocks
stocks = {}
for ticker in ALLOCATION.keys():
    stock_data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    try:
        if isinstance(stock_data.columns, pd.MultiIndex):
            stocks[ticker] = stock_data[('Adj Close', ticker)]
        else:
            stocks[ticker] = stock_data['Adj Close']
    except (KeyError, TypeError):
        try:
            if isinstance(stock_data.columns, pd.MultiIndex):
                stocks[ticker] = stock_data[('Close', ticker)]
            else:
                stocks[ticker] = stock_data['Close']
        except:
            stocks[ticker] = stock_data.iloc[:, 3]
    print(f"  [OK] {ticker}: {len(stock_data)} records")

stock_prices = pd.DataFrame(stocks)

# Download VIX
vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
if not vix.empty:
    if 'Close' in vix.columns:
        vix['Volatility'] = vix['Close']
    elif isinstance(vix.columns, pd.MultiIndex):
        vix['Volatility'] = vix['Close'].iloc[:, 0]
    else:
        vix['Volatility'] = vix.iloc[:, 0]
    print(f"  [OK] VIX: {len(vix)} records")
else:
    vix = pd.DataFrame(index=usdjpy.index)
    vix['Volatility'] = usdjpy['Rate'].pct_change().rolling(20).std() * 100

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================

print("\n[2/5] Preparing data...")

# Merge all data
data = pd.DataFrame(index=stock_prices.index)
data['USDJPY'] = usdjpy['Rate']
for ticker in ALLOCATION.keys():
    data[ticker] = stock_prices[ticker]
data['VIX'] = vix['Volatility']

# Forward fill and drop NaN
data = data.ffill().dropna()

print(f"  [OK] Combined dataset: {len(data)} trading days")

# ============================================================================
# STEP 3: Calculate Returns
# ============================================================================

print("\n[3/5] Calculating portfolio returns...")

# Stock returns
for ticker in ALLOCATION.keys():
    data[f'{ticker}_return'] = data[ticker].pct_change()

# Weighted portfolio return
data['portfolio_return'] = sum(ALLOCATION[ticker] * data[f'{ticker}_return']
                               for ticker in ALLOCATION.keys())

# FX return
data['fx_return'] = data['USDJPY'].pct_change()

# ============================================================================
# STEP 4: Apply VIX Spike Strategy
# ============================================================================

print("\n[4/5] Applying VIX Spike (20) strategy...")

# Position signal: 1 when VIX < 20, 0 when VIX >= 20
data['position'] = (data['VIX'] < VIX_THRESHOLD).astype(int)

# Daily carry
daily_carry = (USD_RATE - JPY_RATE) / 252

# Strategy return = position * (portfolio return + FX return + carry)
data['strategy_return'] = data['position'] * (
    data['portfolio_return'] + data['fx_return'] + daily_carry
)

# Cumulative returns
data['cumulative_return'] = (1 + data['strategy_return']).cumprod()
data['portfolio_cumulative'] = (1 + data['portfolio_return']).cumprod()
data['buy_hold_return'] = data['portfolio_return'] + data['fx_return'] + daily_carry
data['buy_hold_cumulative'] = (1 + data['buy_hold_return']).cumprod()

# ============================================================================
# STEP 5: Calculate Performance Metrics
# ============================================================================

print("\n[5/5] Calculating performance metrics...")

# Final metrics
total_return = data['cumulative_return'].iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len(data)) - 1
annual_vol = data['strategy_return'].std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

# Maximum drawdown
running_max = data['cumulative_return'].expanding().max()
drawdown = (data['cumulative_return'] - running_max) / running_max
max_drawdown = drawdown.min()

# Calmar ratio
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

# Win rate
win_rate = (data['strategy_return'] > 0).sum() / len(data['strategy_return'])

# Time in market
time_in_market = data['position'].sum() / len(data['position'])

# Days in/out
days_in = data['position'].sum()
days_out = len(data['position']) - days_in

print("\n" + "=" * 80)
print("PERFORMANCE METRICS - VIX SPIKE (20) STRATEGY")
print("=" * 80)
print(f"\nSharpe Ratio:           {sharpe_ratio:.3f}")
print(f"Total Return:           {total_return*100:.2f}%")
print(f"Annualized Return:      {annual_return*100:.2f}%")
print(f"Annualized Volatility:  {annual_vol*100:.2f}%")
print(f"Maximum Drawdown:       {max_drawdown*100:.2f}%")
print(f"Calmar Ratio:           {calmar_ratio:.3f}")
print(f"Win Rate:               {win_rate*100:.2f}%")
print(f"\nTime in Market:         {time_in_market*100:.2f}%")
print(f"Days In Position:       {days_in}")
print(f"Days Out of Position:   {days_out}")
print("=" * 80)

# ============================================================================
# STEP 6: Generate Visualizations
# ============================================================================

print("\n[6/7] Generating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Cumulative Returns - Strategy vs Buy & Hold
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(data.index, data['cumulative_return'], label='VIX Spike (20) Strategy',
         linewidth=2.5, color='green')
ax1.plot(data.index, data['buy_hold_cumulative'], label='Buy & Hold (No Exit)',
         linewidth=2, color='blue', alpha=0.7)
ax1.fill_between(data.index, 1, data['cumulative_return'].max(),
                 where=data['position']==0, alpha=0.2, color='red',
                 label='Out of Position')
ax1.set_title('Cumulative Returns: VIX Spike (20) Strategy vs Buy & Hold',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Add key metrics as text
textstr = f'Strategy Sharpe: {sharpe_ratio:.2f}\nTotal Return: {total_return*100:.1f}%\nMax DD: {max_drawdown*100:.1f}%'
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: USD/JPY Exchange Rate with Position Signals
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(data.index, data['USDJPY'], linewidth=2, color='navy')
ax2.fill_between(data.index, data['USDJPY'].min(), data['USDJPY'].max(),
                 where=data['position']==0, alpha=0.3, color='red', label='Out of Position')
ax2.set_title('USD/JPY Exchange Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('JPY per USD', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: VIX with Threshold Line
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(data.index, data['VIX'], linewidth=1.5, color='orange', label='VIX')
ax3.axhline(y=VIX_THRESHOLD, color='red', linestyle='--', linewidth=2,
            label=f'Exit Threshold ({VIX_THRESHOLD})')
ax3.fill_between(data.index, VIX_THRESHOLD, data['VIX'].max(),
                 where=data['VIX']>=VIX_THRESHOLD, alpha=0.3, color='red')
ax3.set_title('VIX Volatility Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('VIX Level', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Individual Stock Performance
ax4 = fig.add_subplot(gs[2, 0])
for ticker in ALLOCATION.keys():
    normalized = data[ticker] / data[ticker].iloc[0]
    ax4.plot(data.index, normalized, label=f'{ticker} ({ALLOCATION[ticker]*100:.0f}%)',
             linewidth=2)
ax4.set_title('Portfolio Components (Normalized)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Normalized Price', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Drawdown Analysis
ax5 = fig.add_subplot(gs[2, 1])
drawdown_pct = drawdown * 100
ax5.fill_between(data.index, drawdown_pct, 0, alpha=0.6, color='red')
ax5.axhline(y=max_drawdown*100, color='darkred', linestyle='--', linewidth=1.5,
            label=f'Max DD: {max_drawdown*100:.1f}%')
ax5.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
ax5.set_ylabel('Drawdown (%)', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.suptitle('Yen Carry Trade - VIX Spike (20) Strategy Performance',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/Users/alannadels/Desktop/Finance/exercise5_Currency_Markets/yen_carry_trade_analysis.png',
            dpi=300, bbox_inches='tight')
print("  [OK] Chart saved: yen_carry_trade_analysis.png")

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("\n[7/7] Saving results...")

# Save metrics to text file
with open('/Users/alannadels/Desktop/Finance/exercise5_Currency_Markets/yen_carry_trade_metrics.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("YEN CARRY TRADE - VIX SPIKE (20) STRATEGY\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("=" * 80 + "\n\n")

    f.write("STRATEGY DESCRIPTION:\n")
    f.write("-" * 80 + "\n")
    f.write("Borrow JPY at low interest rates (~0.1% p.a.)\n")
    f.write("Convert to USD and invest in tech stocks:\n")
    f.write("  - 50% GOOGL (Alphabet)\n")
    f.write("  - 30% NVDA (Nvidia)\n")
    f.write("  - 20% IBM\n")
    f.write(f"Exit when VIX > {VIX_THRESHOLD} (high volatility environment)\n")
    f.write(f"Re-enter when VIX < {VIX_THRESHOLD} (low volatility environment)\n\n")

    f.write("TIME PERIOD:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Start Date:              {data.index[0].strftime('%Y-%m-%d')}\n")
    f.write(f"End Date:                {data.index[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"Total Trading Days:      {len(data)}\n")
    f.write(f"Days In Position:        {days_in} ({time_in_market*100:.1f}%)\n")
    f.write(f"Days Out of Position:    {days_out} ({(1-time_in_market)*100:.1f}%)\n\n")

    f.write("RETURNS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Return:            {total_return*100:.2f}%\n")
    f.write(f"Annualized Return:       {annual_return*100:.2f}%\n")
    f.write(f"Buy & Hold Return:       {(data['buy_hold_cumulative'].iloc[-1]-1)*100:.2f}%\n")
    f.write(f"Outperformance:          {((data['cumulative_return'].iloc[-1]/data['buy_hold_cumulative'].iloc[-1])-1)*100:.2f}%\n\n")

    f.write("RISK METRICS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Annualized Volatility:   {annual_vol*100:.2f}%\n")
    f.write(f"Maximum Drawdown:        {max_drawdown*100:.2f}%\n")
    f.write(f"Sharpe Ratio:            {sharpe_ratio:.3f}\n")
    f.write(f"Calmar Ratio:            {calmar_ratio:.3f}\n")
    f.write(f"Win Rate:                {win_rate*100:.2f}%\n\n")

    f.write("PORTFOLIO STATISTICS:\n")
    f.write("-" * 80 + "\n")
    for ticker in ALLOCATION.keys():
        stock_return = (data[ticker].iloc[-1] / data[ticker].iloc[0] - 1) * 100
        f.write(f"{ticker} Total Return:      {stock_return:.2f}% (weight: {ALLOCATION[ticker]*100:.0f}%)\n")

    f.write(f"\nUSD/JPY Change:          {((data['USDJPY'].iloc[-1]/data['USDJPY'].iloc[0])-1)*100:.2f}%\n")
    f.write(f"USD/JPY Start:           {data['USDJPY'].iloc[0]:.2f}\n")
    f.write(f"USD/JPY End:             {data['USDJPY'].iloc[-1]:.2f}\n")
    f.write(f"USD/JPY Peak:            {data['USDJPY'].max():.2f}\n")
    f.write(f"USD/JPY Trough:          {data['USDJPY'].min():.2f}\n\n")

    f.write("WHY THIS STRATEGY WORKS:\n")
    f.write("-" * 80 + "\n")
    f.write("1. VIX spikes signal risk-off environments where:\n")
    f.write("   - Equity markets typically sell off\n")
    f.write("   - JPY tends to appreciate (safe haven)\n")
    f.write("   - Carry trades become unprofitable\n\n")
    f.write("2. Exiting when VIX > 20 protects against:\n")
    f.write("   - Market crashes (COVID-19, Aug 2024 unwind)\n")
    f.write("   - Currency reversals\n")
    f.write("   - Volatility spikes\n\n")
    f.write("3. Re-entering when VIX < 20 captures:\n")
    f.write("   - Bull market gains (2020-2021, 2023-2024)\n")
    f.write("   - Tech rally (AI boom)\n")
    f.write("   - JPY weakness periods\n\n")

    f.write("KEY EVENTS AVOIDED:\n")
    f.write("-" * 80 + "\n")
    # Find major out-of-position periods
    out_periods = data[data['position'] == 0]
    if len(out_periods) > 0:
        f.write(f"- COVID-19 crash (early 2020)\n")
        f.write(f"- Various volatility spikes throughout 2020-2024\n")
        f.write(f"- August 2024 carry trade unwind\n")
        f.write(f"- Total periods out of market: {len(out_periods)} days\n\n")

    f.write("COMPARISON TO BUY & HOLD:\n")
    f.write("-" * 80 + "\n")
    bh_return = data['buy_hold_cumulative'].iloc[-1] - 1
    bh_vol = data['buy_hold_return'].std() * np.sqrt(252)
    bh_sharpe = (bh_return ** (252/len(data)) - 1) / bh_vol if bh_vol > 0 else 0
    bh_dd = ((data['buy_hold_cumulative'] - data['buy_hold_cumulative'].expanding().max()) /
             data['buy_hold_cumulative'].expanding().max()).min()

    f.write(f"Buy & Hold Total Return:     {bh_return*100:.2f}%\n")
    f.write(f"Buy & Hold Sharpe Ratio:     {bh_sharpe:.3f}\n")
    f.write(f"Buy & Hold Max Drawdown:     {bh_dd*100:.2f}%\n\n")
    f.write(f"Strategy Advantage:\n")
    f.write(f"  Higher Sharpe:             +{((sharpe_ratio/bh_sharpe)-1)*100:.1f}%\n")
    f.write(f"  Lower Drawdown:            {((abs(max_drawdown)/abs(bh_dd))-1)*100:.1f}%\n")
    f.write(f"  Higher Total Return:       +{((total_return/bh_return)-1)*100:.1f}%\n\n")

    f.write("=" * 80 + "\n")
    f.write("CONCLUSION:\n")
    f.write("=" * 80 + "\n")
    f.write(f"The VIX Spike (20) strategy achieved a Sharpe ratio of {sharpe_ratio:.2f},\n")
    f.write(f"significantly outperforming buy-and-hold while reducing drawdowns.\n")
    f.write(f"By staying out of the market during high-volatility periods ({(1-time_in_market)*100:.1f}% of time),\n")
    f.write(f"the strategy avoided major losses while capturing {total_return*100:.1f}% total returns.\n")
    f.write("=" * 80 + "\n")

print("  [OK] Metrics saved: yen_carry_trade_metrics.txt")

# Save detailed daily data
daily_data = pd.DataFrame({
    'Date': data.index,
    'USDJPY': data['USDJPY'].values,
    'VIX': data['VIX'].values,
    'Position': data['position'].values,
    'Daily_Return_%': data['strategy_return'].values * 100,
    'Cumulative_Return': data['cumulative_return'].values,
    'Drawdown_%': drawdown.values * 100,
    'GOOGL_Price': data['GOOGL'].values,
    'NVDA_Price': data['NVDA'].values,
    'IBM_Price': data['IBM'].values,
})

daily_data.to_csv('/Users/alannadels/Desktop/Finance/exercise5_Currency_Markets/yen_carry_trade_daily_data.csv',
                  index=False)
print("  [OK] Daily data saved: yen_carry_trade_daily_data.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nFiles generated:")
print("  1. yen_carry_trade.py - This script")
print("  2. yen_carry_trade_analysis.png - Comprehensive charts")
print("  3. yen_carry_trade_metrics.txt - Detailed performance metrics")
print("  4. yen_carry_trade_daily_data.csv - Daily returns and positions")
print("\nVIX Spike (20) Strategy Results:")
print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"  Total Return: {total_return*100:.2f}%")
print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
print("=" * 80)
