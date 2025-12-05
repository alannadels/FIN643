"""
Configuration for Options Strategy
"""

# Tickers
EQUITY_TICKER = 'CVX'
FIXED_INCOME_TICKER = 'HYG'

# Date Range
START_DATE = '2021-01-01'
END_DATE = '2024-12-31'

# Option Parameters
RISK_FREE_RATE = 0.02
OPTION_EXPIRY_DAYS = 21

# Strategy IDs
STRATEGIES = {
    0: 'Covered Call',
    1: 'Protective Put',
    2: 'Long Straddle',
    3: 'Long Strangle'
}

# Weight Classes for equity allocation
WEIGHT_CLASSES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Trading Signal Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL_LOW = 45
RSI_NEUTRAL_HIGH = 55

# Volatility Thresholds (annualized)
VOL_LOW = 0.15
VOL_HIGH = 0.30

# Trend Thresholds
TREND_UP = 0.02
TREND_DOWN = -0.02

# Backtest Parameters
INITIAL_CAPITAL = 100000
REBALANCE_FREQUENCY = 21
TRANSACTION_COST = 0.0002  # 0.02% per trade (more realistic for ETFs)

# File Paths
DATA_PATH = 'data'
RESULTS_PATH = 'results'
