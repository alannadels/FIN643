"""
Black-Scholes option pricing for simulated options data
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import config


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annual)

    Returns:
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes put option price

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annual)

    Returns:
        Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks

    Returns:
        dict with delta, gamma, theta, vega
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:  # put
        delta = -norm.cdf(-d1)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:  # put
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))

    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Daily theta
        'vega': vega / 100  # Per 1% change in volatility
    }


def generate_simulated_options(stock_data):
    """
    Generate simulated options data using Black-Scholes

    For each trading day, generate:
    - Calls and puts at various strikes
    - ~30 day expiration
    - Greeks and implied volatility
    """
    print("\nGenerating simulated options using Black-Scholes...")

    options_list = []
    dte = 30  # Days to expiry
    T = dte / 252  # Time in years

    for date, row in stock_data.iterrows():
        S = row['Close']
        sigma = row['Volatility']  # Use historical volatility as IV proxy

        # Skip if missing data
        if np.isnan(S) or np.isnan(sigma) or sigma <= 0:
            continue

        # Generate strikes around current price
        # Calls: 0.25-0.35 delta → roughly 105-110% of spot
        # Puts: -0.25 to -0.35 delta → roughly 90-95% of spot
        call_strikes = [S * mult for mult in [1.05, 1.075, 1.10, 1.125]]
        put_strikes = [S * mult for mult in [0.90, 0.925, 0.95, 0.975]]

        # Generate calls
        for K in call_strikes:
            price = black_scholes_call(S, K, T, config.RISK_FREE_RATE, sigma)
            greeks = calculate_greeks(S, K, T, config.RISK_FREE_RATE, sigma, 'call')

            # Simulate bid-ask spread (0.5% of price)
            spread = price * 0.005
            bid = price - spread
            ask = price + spread

            options_list.append({
                'date': date,
                'exdate': date + pd.Timedelta(days=dte),
                'cp_flag': 'C',
                'strike_price': K,
                'best_bid': max(bid, 0.01),
                'best_offer': max(ask, 0.02),
                'mid_price': price,
                'impl_volatility': sigma,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'days_to_expiry': dte,
                'volume': 1000,  # Dummy
                'open_interest': 1000  # Dummy
            })

        # Generate puts
        for K in put_strikes:
            price = black_scholes_put(S, K, T, config.RISK_FREE_RATE, sigma)
            greeks = calculate_greeks(S, K, T, config.RISK_FREE_RATE, sigma, 'put')

            # Simulate bid-ask spread
            spread = price * 0.005
            bid = price - spread
            ask = price + spread

            options_list.append({
                'date': date,
                'exdate': date + pd.Timedelta(days=dte),
                'cp_flag': 'P',
                'strike_price': K,
                'best_bid': max(bid, 0.01),
                'best_offer': max(ask, 0.02),
                'mid_price': price,
                'impl_volatility': sigma,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'days_to_expiry': dte,
                'volume': 1000,  # Dummy
                'open_interest': 1000  # Dummy
            })

    options_df = pd.DataFrame(options_list)
    # Convert timezone-aware datetimes to timezone-naive
    options_df['date'] = pd.to_datetime(options_df['date'], utc=True).dt.tz_localize(None)
    options_df['exdate'] = pd.to_datetime(options_df['exdate'], utc=True).dt.tz_localize(None)

    print(f"Generated {len(options_df)} simulated option quotes")
    print(f"  Calls: {len(options_df[options_df['cp_flag'] == 'C'])}")
    print(f"  Puts: {len(options_df[options_df['cp_flag'] == 'P'])}")

    return options_df
