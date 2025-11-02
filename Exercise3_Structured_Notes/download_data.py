"""
Download historical data for NVDA and URA
"""
import yfinance as yf
import pandas as pd
from datetime import datetime

# Download data from 2020 onwards (5 years of history)
# This gives us enough data to backtest and understand volatility patterns
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

print("Downloading NVDA data...")
nvda = yf.Ticker("NVDA")
nvda_data = nvda.history(start=start_date, end=end_date)

print("Downloading URA data...")
ura = yf.Ticker("URA")
ura_data = ura.history(start=start_date, end=end_date)

# Remove timezone if present
if hasattr(nvda_data.index, 'tz') and nvda_data.index.tz is not None:
    nvda_data.index = nvda_data.index.tz_localize(None)

if hasattr(ura_data.index, 'tz') and ura_data.index.tz is not None:
    ura_data.index = ura_data.index.tz_localize(None)

# Save to CSV
nvda_data.to_csv('nvda_data.csv')
ura_data.to_csv('ura_data.csv')

print(f"\nNVDA data: {len(nvda_data)} days from {nvda_data.index[0]} to {nvda_data.index[-1]}")
print(f"URA data: {len(ura_data)} days from {ura_data.index[0]} to {ura_data.index[-1]}")

# Display summary statistics
print("\n" + "="*60)
print("NVDA Summary Statistics:")
print("="*60)
print(nvda_data['Close'].describe())

print("\n" + "="*60)
print("URA Summary Statistics:")
print("="*60)
print(ura_data['Close'].describe())

# Calculate some basic metrics
nvda_returns = nvda_data['Close'].pct_change().dropna()
ura_returns = ura_data['Close'].pct_change().dropna()

print("\n" + "="*60)
print("Historical Performance Metrics:")
print("="*60)
print(f"NVDA Annualized Return: {nvda_returns.mean() * 252 * 100:.2f}%")
print(f"NVDA Annualized Volatility: {nvda_returns.std() * (252**0.5) * 100:.2f}%")
print(f"\nURA Annualized Return: {ura_returns.mean() * 252 * 100:.2f}%")
print(f"URA Annualized Volatility: {ura_returns.std() * (252**0.5) * 100:.2f}%")

# Calculate correlation
aligned_nvda = nvda_data['Close'].reindex(ura_data.index)
aligned_ura = ura_data['Close']
correlation = aligned_nvda.pct_change().corr(aligned_ura.pct_change())
print(f"\nCorrelation between NVDA and URA: {correlation:.3f}")

print("\n" + "="*60)
print("Data downloaded and saved successfully!")
print("="*60)
