"""
Helper functions for Sharpe ratio calculation and results printing
"""
import numpy as np
import pandas as pd

def calculate_sharpe_from_results(results_df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """
    Calculate Sharpe ratio from backtest results

    Args:
        results_df: DataFrame with 'total_return' column (3-year returns)
        risk_free_rate: Annual risk-free rate (default 4%)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(results_df) == 0:
        return 0.0

    # Convert 3-year returns to annualized returns
    annual_returns = (1 + results_df['total_return']) ** (1/3) - 1

    mean_annual_return = annual_returns.mean()
    std_annual_return = annual_returns.std()

    if std_annual_return == 0:
        return 0.0

    sharpe = (mean_annual_return - risk_free_rate) / std_annual_return

    return sharpe


def print_final_results(sharpe: float, config: dict, results_df: pd.DataFrame):
    """
    Print formatted summary of optimization results

    Args:
        sharpe: Sharpe ratio achieved
        config: Configuration dictionary
        results_df: Backtest results DataFrame
    """
    print("\n" + "="*80)
    print("FINAL OPTIMIZATION RESULTS")
    print("="*80)

    print(f"\nConfiguration: {config.get('name', 'Optimal')}")
    print(f"\nSharpe Ratio: {sharpe:.4f}")

    print("\nParameters:")
    for key, value in config.items():
        if key != 'name':
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    # Calculate summary statistics
    annual_returns = (1 + results_df['total_return']) ** (1/3) - 1

    print("\nPerformance Metrics:")
    print(f"  Mean 3-Year Return: {results_df['total_return'].mean()*100:.2f}%")
    print(f"  Median 3-Year Return: {results_df['total_return'].median()*100:.2f}%")
    print(f"  Best 3-Year Return: {results_df['total_return'].max()*100:.2f}%")
    print(f"  Worst 3-Year Return: {results_df['total_return'].min()*100:.2f}%")
    print(f"  Std Dev (3-Year): {results_df['total_return'].std()*100:.2f}%")
    print()
    print(f"  Mean Annual Return: {annual_returns.mean()*100:.2f}%")
    print(f"  Annual Std Dev: {annual_returns.std()*100:.2f}%")

    print("\nStructure Performance:")
    print(f"  Autocall Rate: {results_df['autocalled'].sum() / len(results_df) * 100:.1f}%")
    print(f"  Average Holding Period: {results_df['days_held'].mean():.0f} days ({results_df['days_held'].mean()/252:.2f} years)")
    print(f"  Average Coupons Received: ${results_df['coupons_received'].mean():,.2f}")
    print(f"  Average Basket Return at Exit: {results_df['basket_return'].mean()*100:.2f}%")

    print("\n" + "="*80)
