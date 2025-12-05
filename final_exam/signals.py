"""
Trading Signals and Technical Indicators for Strategy Selection
"""

import numpy as np
import pandas as pd
from option_pricing import calculate_iv_rank, calculate_iv_percentile
import config


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate annualized volatility"""
    returns = np.log(prices / prices.shift(1))
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()


def calculate_trend(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.Series:
    """Calculate trend strength: (SMA_short - SMA_long) / SMA_long"""
    sma_short = calculate_sma(prices, short_window)
    sma_long = calculate_sma(prices, long_window)
    return (sma_short - sma_long) / sma_long


def calculate_momentum(prices: pd.Series, period: int = 5) -> pd.Series:
    """Calculate price momentum (return over period)"""
    return prices.pct_change(period)


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    std = prices.rolling(window=window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    # Percent B: where price sits within bands (0 = lower, 1 = upper)
    percent_b = (prices - lower) / (upper - lower)

    return upper, lower, percent_b


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def generate_all_signals(equity_data: pd.DataFrame, fi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all trading signals for strategy selection

    Returns DataFrame with all signals aligned to common dates
    """
    # Align data
    common_dates = equity_data.index.intersection(fi_data.index)
    equity = equity_data.loc[common_dates]
    fi = fi_data.loc[common_dates]

    signals = pd.DataFrame(index=common_dates)

    # Equity Signals
    signals['Equity_Close'] = equity['Close']
    signals['Equity_RSI'] = calculate_rsi(equity['Close'])
    signals['Equity_Volatility'] = calculate_volatility(equity['Close'])
    signals['Equity_Trend'] = calculate_trend(equity['Close'])
    signals['Equity_Momentum_5d'] = calculate_momentum(equity['Close'], 5)
    signals['Equity_Momentum_21d'] = calculate_momentum(equity['Close'], 21)

    # Bollinger Bands
    _, _, signals['Equity_BB_PercentB'] = calculate_bollinger_bands(equity['Close'])

    # ATR for Equity
    signals['Equity_ATR'] = calculate_atr(equity['High'], equity['Low'], equity['Close'])
    signals['Equity_ATR_Percent'] = signals['Equity_ATR'] / equity['Close'] * 100

    # MACD
    macd, macd_signal, macd_hist = calculate_macd(equity['Close'])
    signals['Equity_MACD'] = macd
    signals['Equity_MACD_Signal'] = macd_signal
    signals['Equity_MACD_Hist'] = macd_hist

    # IV Rank (using realized vol as proxy)
    vol_252 = signals['Equity_Volatility'].rolling(252, min_periods=60).apply(
        lambda x: calculate_iv_rank(x.iloc[-1], x.iloc[:-1]) if len(x) > 1 else 50
    )
    signals['Equity_IV_Rank'] = vol_252.fillna(50)

    # Fixed Income Signals
    signals['FI_Close'] = fi['Close']
    signals['FI_RSI'] = calculate_rsi(fi['Close'])
    signals['FI_Volatility'] = calculate_volatility(fi['Close'])
    signals['FI_Momentum_5d'] = calculate_momentum(fi['Close'], 5)

    # Cross-asset signals
    signals['Correlation_60d'] = equity['Close'].rolling(60).corr(fi['Close'])

    # Volatility regime (0=Low, 1=Medium, 2=High)
    signals['Vol_Regime'] = pd.cut(
        signals['Equity_Volatility'],
        bins=[0, config.VOL_LOW, config.VOL_HIGH, np.inf],
        labels=[0, 1, 2]
    ).astype(float)

    # Drop NaN rows
    signals = signals.dropna()

    print(f"Generated signals: {len(signals)} days")
    print(f"Date range: {signals.index[0]} to {signals.index[-1]}")

    return signals


def get_signal_summary(signals: pd.DataFrame, idx: int) -> dict:
    """Get a summary of all signals at a specific index for strategy decision"""
    row = signals.iloc[idx]

    return {
        'equity_price': row['Equity_Close'],
        'equity_rsi': row['Equity_RSI'],
        'equity_volatility': row['Equity_Volatility'],
        'equity_trend': row['Equity_Trend'],
        'equity_momentum_5d': row['Equity_Momentum_5d'],
        'equity_momentum_21d': row['Equity_Momentum_21d'],
        'equity_bb_percentb': row['Equity_BB_PercentB'],
        'equity_atr_percent': row['Equity_ATR_Percent'],
        'equity_macd_hist': row['Equity_MACD_Hist'],
        'equity_iv_rank': row['Equity_IV_Rank'],
        'fi_price': row['FI_Close'],
        'fi_rsi': row['FI_RSI'],
        'fi_volatility': row['FI_Volatility'],
        'fi_momentum_5d': row['FI_Momentum_5d'],
        'correlation': row['Correlation_60d'],
        'vol_regime': row['Vol_Regime'],
        'macd_hist': row['Equity_MACD_Hist']  # Alias for convenience
    }
