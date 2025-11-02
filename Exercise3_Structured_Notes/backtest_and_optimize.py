"""
Backtesting framework for structured note using rolling windows
"""
import pandas as pd
from structured_note import AIEnergyStructuredNote

def rolling_window_backtest(
    note: AIEnergyStructuredNote,
    nvda_prices: pd.Series,
    ura_prices: pd.Series,
    window_step: int = 126
) -> pd.DataFrame:
    """
    Backtest the structured note using rolling 3-year windows

    Args:
        note: Configured AIEnergyStructuredNote instance
        nvda_prices: NVDA price series
        ura_prices: URA price series
        window_step: Number of trading days between window starts (default 126 = ~6 months)

    Returns:
        DataFrame with results from each window
    """
    results = []
    window_size = note.maturity_days  # 3 years = 756 trading days

    # Create rolling windows
    max_start_idx = len(nvda_prices) - window_size - 1

    for start_idx in range(0, max_start_idx, window_step):
        # Simulate note for this window
        result = note.simulate_note(nvda_prices, ura_prices, start_idx)

        # Add metadata
        result['start_date'] = nvda_prices.index[start_idx]
        result['window_idx'] = start_idx // window_step

        results.append(result)

    return pd.DataFrame(results)
