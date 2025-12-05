"""
Black-Scholes Option Pricing and Greeks
"""

import numpy as np
from scipy.stats import norm
import config


def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate option delta"""
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def calculate_gamma(S, K, T, r, sigma):
    """Calculate option gamma (same for call and put)"""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calculate_theta(S, K, T, r, sigma, option_type='call'):
    """Calculate option theta (daily)"""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    # Return daily theta
    return (term1 + term2) / 252


def calculate_vega(S, K, T, r, sigma):
    """Calculate option vega (per 1% vol change)"""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all Greeks for an option"""
    return {
        'delta': calculate_delta(S, K, T, r, sigma, option_type),
        'gamma': calculate_gamma(S, K, T, r, sigma),
        'theta': calculate_theta(S, K, T, r, sigma, option_type),
        'vega': calculate_vega(S, K, T, r, sigma)
    }


def calculate_iv_rank(current_vol, vol_history):
    """
    Calculate IV Rank: where current vol sits in the past year's range
    0 = lowest vol, 100 = highest vol
    """
    if len(vol_history) == 0:
        return 50

    min_vol = np.min(vol_history)
    max_vol = np.max(vol_history)

    if max_vol == min_vol:
        return 50

    return (current_vol - min_vol) / (max_vol - min_vol) * 100


def calculate_iv_percentile(current_vol, vol_history):
    """
    Calculate IV Percentile: % of days vol was lower than current
    """
    if len(vol_history) == 0:
        return 50

    return np.sum(vol_history < current_vol) / len(vol_history) * 100
