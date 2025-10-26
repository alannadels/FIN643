"""
Utility functions for options trading strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


def load_data():
    """Load stock and options data from disk"""
    data_dir = Path(config.DATA_PATH)
    data_dir.mkdir(exist_ok=True)

    # Load stock data
    stock_path = data_dir / f'{config.TICKER}_stock_data.csv'

    # Download if doesn't exist
    if not stock_path.exists():
        print(f"Downloading {config.TICKER} stock data from Yahoo Finance...")
        import yfinance as yf

        ticker = yf.Ticker(config.TICKER)
        stock_data = ticker.history(start=config.START_DATE, end=config.END_DATE_TEST)

        # Add technical indicators
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

        # RSI
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        # Volatility (20-day)
        stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        # Volume ratio
        stock_data['Volume_Ratio'] = stock_data['Volume'] / stock_data['Volume'].rolling(window=20).mean()

        # 5-day return
        stock_data['Return_5d'] = stock_data['Close'].pct_change(5)

        # Distance from 52-week high
        stock_data['Dist_from_High'] = (stock_data['Close'] - stock_data['Close'].rolling(window=252).max()) / stock_data['Close'].rolling(window=252).max()

        # Drop NaN rows
        stock_data = stock_data.dropna()

        # Remove timezone to avoid comparison issues
        if hasattr(stock_data.index, 'tz') and stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)

        # Save
        stock_data.to_csv(stock_path)
        print(f"Saved stock data to {stock_path}")
    else:
        stock_data = pd.read_csv(stock_path, index_col=0, parse_dates=True)

    # Load options data if available
    options_path = data_dir / f'{config.TICKER}_options_data.csv'
    if options_path.exists():
        options_data = pd.read_csv(options_path, parse_dates=['date', 'exdate'])
        print(f"Loaded {len(options_data)} option quotes")
    else:
        # Generate simulated options using Black-Scholes
        print("No WRDS options data found - generating simulated options using Black-Scholes")
        from black_scholes import generate_simulated_options
        options_data = generate_simulated_options(stock_data)

    print(f"Loaded {len(stock_data)} days of stock data")

    return stock_data, options_data


def split_data(stock_data, options_data=None):
    """Split data into train, validation, and test sets"""

    # Convert string dates to proper datetime for comparison
    start_date = pd.to_datetime(config.START_DATE)
    end_train_date = pd.to_datetime(config.END_DATE_TRAIN)

    # Test set: everything after training period
    test_date = end_train_date + pd.Timedelta(days=1)
    test_mask = stock_data.index >= test_date
    test_stock = stock_data[test_mask].copy()

    # Training pool: START_DATE to END_DATE_TRAIN
    train_mask = (stock_data.index >= start_date) & (stock_data.index <= end_train_date)
    train_pool = stock_data[train_mask].copy()

    # Validation: random 20% of training pool dates
    np.random.seed(42)
    train_dates = train_pool.index.tolist()
    n_val = int(len(train_dates) * config.VALIDATION_RATIO)
    val_dates = np.random.choice(train_dates, size=n_val, replace=False)

    val_stock = train_pool.loc[val_dates].copy().sort_index()
    train_stock = train_pool.drop(val_dates).copy()

    print(f"\nData split:")
    print(f"  Training: {len(train_stock)} days ({train_stock.index[0]} to {train_stock.index[-1]})")
    print(f"  Validation: {len(val_stock)} days (random sample)")
    if len(test_stock) > 0:
        print(f"  Test: {len(test_stock)} days ({test_stock.index[0]} to {test_stock.index[-1]})")
    else:
        print(f"  Test: {len(test_stock)} days (NO DATA - check END_DATE_TEST in config)")
        raise ValueError(f"No test data found! Check that END_DATE_TEST ({config.END_DATE_TEST}) has data available.")

    # Split options data if available
    if options_data is not None:
        test_options = options_data[options_data['date'] >= test_date].copy()
        train_options = options_data[
            (options_data['date'] >= start_date) &
            (options_data['date'] <= end_train_date)
        ].copy()
        val_options = train_options[train_options['date'].isin(val_dates)].copy()
        train_options = train_options[~train_options['date'].isin(val_dates)].copy()

        print(f"  Training options: {len(train_options)} quotes")
        print(f"  Validation options: {len(val_options)} quotes")
        print(f"  Test options: {len(test_options)} quotes")

        return (train_stock, val_stock, test_stock,
                train_options, val_options, test_options)
    else:
        return train_stock, val_stock, test_stock, None, None, None


def get_option_for_strategy(options_data, current_date, stock_price, strategy='covered_call'):
    """
    Find appropriate option for given strategy on given date

    Args:
        options_data: DataFrame of historical options
        current_date: Date to find option for
        stock_price: Current stock price
        strategy: 'covered_call', 'cash_put', 'protective_put', 'collar_call', 'collar_put'

    Returns:
        Option data dict or None if not found
    """
    if options_data is None:
        return None

    # Filter options for this date
    date_options = options_data[options_data['date'] == current_date].copy()

    if len(date_options) == 0:
        return None

    # Filter by days to expiry (around 30 days)
    date_options = date_options[
        (date_options['days_to_expiry'] >= 25) &
        (date_options['days_to_expiry'] <= 35)
    ].copy()

    if len(date_options) == 0:
        return None

    # Select based on strategy
    if strategy == 'straddle_call' or strategy == 'straddle_put':
        # ATM options for straddle - delta around 0.50 for calls, -0.50 for puts
        if strategy == 'straddle_call':
            calls = date_options[date_options['cp_flag'] == 'C'].copy()
            # Find closest to ATM (delta ~0.50)
            calls['delta_diff'] = np.abs(calls['delta'] - 0.50)
            if len(calls) == 0:
                return None
            best_option = calls.nsmallest(1, 'delta_diff').iloc[0]
        else:  # straddle_put
            puts = date_options[date_options['cp_flag'] == 'P'].copy()
            # Find closest to ATM (delta ~-0.50)
            puts['delta_diff'] = np.abs(puts['delta'] + 0.50)
            if len(puts) == 0:
                return None
            best_option = puts.nsmallest(1, 'delta_diff').iloc[0]

        return {
            'type': 'call' if strategy == 'straddle_call' else 'put',
            'strike': best_option['strike_price'],
            'bid': best_option['best_bid'],
            'ask': best_option['best_offer'],
            'mid': best_option['mid_price'],
            'delta': best_option['delta'],
            'gamma': best_option['gamma'],
            'theta': best_option['theta'],
            'iv': best_option['impl_volatility'],
            'dte': best_option['days_to_expiry']
        }

    elif strategy == 'covered_call' or strategy == 'collar_call':
        # OTM calls with delta 0.25-0.35
        calls = date_options[date_options['cp_flag'] == 'C'].copy()
        calls = calls[
            (calls['delta'] >= config.DELTA_RANGE_CALL[0]) &
            (calls['delta'] <= config.DELTA_RANGE_CALL[1])
        ]

        if len(calls) == 0:
            return None

        # Pick closest to 0.30 delta
        calls['delta_diff'] = np.abs(calls['delta'] - 0.30)
        best_call = calls.nsmallest(1, 'delta_diff').iloc[0]

        return {
            'type': 'call',
            'strike': best_call['strike_price'],
            'bid': best_call['best_bid'],
            'ask': best_call['best_offer'],
            'mid': best_call['mid_price'],
            'delta': best_call['delta'],
            'gamma': best_call['gamma'],
            'theta': best_call['theta'],
            'iv': best_call['impl_volatility'],
            'dte': best_call['days_to_expiry']
        }

    elif strategy == 'cash_put' or strategy == 'protective_put' or strategy == 'collar_put':
        # OTM puts with delta -0.25 to -0.35
        puts = date_options[date_options['cp_flag'] == 'P'].copy()
        puts = puts[
            (puts['delta'] >= config.DELTA_RANGE_PUT[0]) &
            (puts['delta'] <= config.DELTA_RANGE_PUT[1])
        ]

        if len(puts) == 0:
            return None

        # Pick closest to -0.30 delta
        puts['delta_diff'] = np.abs(puts['delta'] + 0.30)
        best_put = puts.nsmallest(1, 'delta_diff').iloc[0]

        return {
            'type': 'put',
            'strike': best_put['strike_price'],
            'bid': best_put['best_bid'],
            'ask': best_put['best_offer'],
            'mid': best_put['mid_price'],
            'delta': best_put['delta'],
            'gamma': best_put['gamma'],
            'theta': best_put['theta'],
            'iv': best_put['impl_volatility'],
            'dte': best_put['days_to_expiry']
        }

    return None


def calculate_sharpe_ratio(returns, risk_free_rate=None):
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate (default from config)

    Returns:
        Sharpe ratio (annualized)
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    returns = np.array(returns)
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)


def normalize_state(state_dict, stock_data):
    """Normalize state features"""
    # Price relative to 200 SMA
    price_norm = state_dict['price'] / state_dict['sma_200'] if state_dict['sma_200'] > 0 else 1.0

    # RSI already 0-100
    rsi_norm = state_dict['rsi'] / 100.0

    # Volatility - normalize by historical mean
    vol_mean = stock_data['Volatility'].mean()
    vol_norm = state_dict['volatility'] / vol_mean if vol_mean > 0 else 1.0

    # Trend
    trend_norm = (state_dict['sma_50'] - state_dict['sma_200']) / state_dict['sma_200'] if state_dict['sma_200'] > 0 else 0.0

    # Volume ratio already normalized
    volume_norm = state_dict['volume_ratio']

    # Price changes
    return_5d_norm = state_dict['return_5d']
    dist_from_high_norm = state_dict['dist_from_high']

    # VIX proxy (using NVDA's volatility as proxy)
    vix_norm = vol_norm

    return np.array([
        price_norm,
        rsi_norm,
        vol_norm,
        trend_norm,
        volume_norm,
        return_5d_norm,
        dist_from_high_norm,
        vix_norm
    ], dtype=np.float32)
