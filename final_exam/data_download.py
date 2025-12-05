"""
Data download module for equity and fixed income assets
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import config


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical data from Yahoo Finance"""
    print(f"Downloading {ticker} data...")

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove timezone
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    print(f"  Downloaded {len(data)} days of data")
    return data


def save_data(data: pd.DataFrame, ticker: str, path: str) -> None:
    """Save data to CSV"""
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = Path(path) / f"{ticker}_data.csv"
    data.to_csv(filepath)
    print(f"  Saved to {filepath}")


def load_data(ticker: str, path: str) -> pd.DataFrame:
    """Load data from CSV"""
    filepath = Path(path) / f"{ticker}_data.csv"
    if filepath.exists():
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded {ticker} data: {len(data)} days")
        return data
    return None


def get_data(ticker: str, start_date: str, end_date: str,
             data_path: str, force_download: bool = False) -> pd.DataFrame:
    """Get data - load from cache or download"""

    if not force_download:
        data = load_data(ticker, data_path)
        if data is not None:
            return data

    data = download_data(ticker, start_date, end_date)
    save_data(data, ticker, data_path)
    return data


if __name__ == '__main__':
    equity = get_data(config.EQUITY_TICKER, config.START_DATE, config.END_DATE,
                      config.DATA_PATH, force_download=True)
    fi = get_data(config.FIXED_INCOME_TICKER, config.START_DATE, config.END_DATE,
                  config.DATA_PATH, force_download=True)

    print(f"\n{config.EQUITY_TICKER}: {equity.index[0]} to {equity.index[-1]}")
    print(f"{config.FIXED_INCOME_TICKER}: {fi.index[0]} to {fi.index[-1]}")
