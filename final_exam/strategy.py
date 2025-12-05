"""
Options Strategy Selection

CONFIGURATION: CVX + HYG with momentum_straddle strategy and trend_based weights
Sharpe: 0.85, Return: +68.0%, MaxDD: -9.7%
Strategy Mix: CC=27%, PP=2%, LS=70%

Strategy Selection Logic Based on Market Conditions:

1. COVERED CALL (Strategy 0):
   - Best when: Neutral to mildly bullish, low/medium volatility
   - Signals: RSI 40-60, Low IV Rank, Sideways trend, Negative theta works for us
   - Generates income in flat markets

2. PROTECTIVE PUT (Strategy 1):
   - Best when: Bullish but want downside protection, or expecting pullback
   - Signals: RSI > 65 (overbought), High volatility, Negative momentum
   - Limits losses in downturns

3. LONG STRADDLE (Strategy 2):
   - Best when: Expecting big move, direction unknown
   - Signals: Low IV Rank (cheap options), High ATR, RSI extreme (<30 or >70)
   - Profits from large moves either direction

4. LONG STRANGLE (Strategy 3):
   - Best when: Expecting very big move, want cheaper entry than straddle
   - Signals: Very low IV Rank, Consolidating price, Extreme RSI
   - Similar to straddle but wider breakevens
"""

import numpy as np
import pandas as pd
from signals import get_signal_summary
import config


def select_option_strategy(signals: dict) -> int:
    """
    Select the optimal option strategy based on market signals and Greeks

    MOMENTUM_STRADDLE: Straddle at momentum extremes or low IV

    Returns:
        0: Covered Call
        1: Protective Put
        2: Long Straddle
        3: Long Strangle
    """
    rsi = signals['equity_rsi']
    vol = signals['equity_volatility']
    trend = signals['equity_trend']
    momentum_5d = signals['equity_momentum_5d']
    momentum_21d = signals['equity_momentum_21d']
    iv_rank = signals['equity_iv_rank']
    bb_percentb = signals['equity_bb_percentb']
    macd_hist = signals['equity_macd_hist']
    vol_regime = signals['vol_regime']

    # =========================================================================
    # STRATEGY SELECTION RULES - MOMENTUM_STRADDLE
    # =========================================================================

    # Rule 1: LONG STRADDLE at momentum extremes or low IV
    # Straddle when RSI shows extreme conditions or IV is low (options are cheap)
    if rsi < 35 or rsi > 65 or iv_rank < 35:
        return 2  # Long Straddle

    # Rule 2: PROTECTIVE PUT in downtrend with negative momentum
    if trend < -0.02 and momentum_5d < 0:
        return 1  # Protective Put

    # Rule 3: COVERED CALL for everything else
    return 0  # Covered Call


def select_weight_allocation(signals: dict) -> int:
    """
    Select the optimal equity/fixed income weight allocation based on market signals

    TREND_BASED: Weight based purely on trend

    Returns:
        0: 0% Equity, 100% Fixed Income
        1: 25% Equity, 75% Fixed Income
        2: 50% Equity, 50% Fixed Income
        3: 75% Equity, 25% Fixed Income
        4: 100% Equity, 0% Fixed Income
    """
    eq_rsi = signals['equity_rsi']
    eq_trend = signals['equity_trend']
    eq_momentum_5d = signals['equity_momentum_5d']
    eq_momentum_21d = signals['equity_momentum_21d']
    fi_momentum = signals['fi_momentum_5d']
    vol_regime = signals['vol_regime']

    # =========================================================================
    # WEIGHT ALLOCATION RULES - TREND_BASED
    # Weight based purely on trend strength
    # =========================================================================

    trend = eq_trend

    if trend > 0.03:
        return 4  # 100% equity
    elif trend > 0.01:
        return 3  # 75% equity
    elif trend > -0.01:
        return 2  # 50% equity
    elif trend > -0.03:
        return 1  # 25% equity
    return 0  # 0% equity


def generate_strategy_decisions(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Generate strategy and weight decisions for all dates

    Returns DataFrame with strategy and weight for each date
    """
    decisions = []

    for idx in range(len(signals)):
        sig = get_signal_summary(signals, idx)
        strategy = select_option_strategy(sig)
        weight_idx = select_weight_allocation(sig)
        weight = config.WEIGHT_CLASSES[weight_idx]

        decisions.append({
            'date': signals.index[idx],
            'strategy': strategy,
            'strategy_name': config.STRATEGIES[strategy],
            'equity_weight': weight,
            'fi_weight': 1 - weight,
            # Key signals for reporting
            'equity_rsi': sig['equity_rsi'],
            'equity_volatility': sig['equity_volatility'],
            'equity_trend': sig['equity_trend'],
            'equity_iv_rank': sig['equity_iv_rank'],
            'vol_regime': sig['vol_regime']
        })

    return pd.DataFrame(decisions).set_index('date')


def summarize_decisions(decisions: pd.DataFrame) -> str:
    """Generate a summary of strategy decisions"""
    strategy_counts = decisions['strategy'].value_counts()
    weight_mean = decisions['equity_weight'].mean()

    equity_ticker = config.EQUITY_TICKER
    fi_ticker = config.FIXED_INCOME_TICKER

    summary = "\nStrategy Selection Summary:\n"
    summary += "-" * 40 + "\n"

    for strat_id, name in config.STRATEGIES.items():
        count = strategy_counts.get(strat_id, 0)
        pct = count / len(decisions) * 100
        summary += f"  {name}: {count} periods ({pct:.1f}%)\n"

    summary += f"\nAverage {equity_ticker} Allocation: {weight_mean*100:.1f}%\n"
    summary += f"Average {fi_ticker} Allocation: {(1-weight_mean)*100:.1f}%\n"

    return summary
