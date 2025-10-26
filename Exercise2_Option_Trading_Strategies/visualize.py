"""
Create visualizations for strategy comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config


def create_visualizations():
    """Create all visualizations from test results"""
    print("Creating visualizations...")

    # Load results
    results_dir = Path(config.RESULTS_PATH)
    results_path = results_dir / 'test_results.csv'

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run evaluate.py first.")
        return

    results_df = pd.read_csv(results_path)
    results_df['Date'] = pd.to_datetime(results_df['Date'])

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 10)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # 1. Cumulative Portfolio Value
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(results_df['Date'], results_df['BuyHold_Value'],
             label='Buy & Hold', linewidth=2, alpha=0.8)
    ax1.plot(results_df['Date'], results_df['RL_Value'],
             label='RL Options Strategy', linewidth=2, alpha=0.8)
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')

    # 2. Cumulative Returns (%)
    ax2 = plt.subplot(2, 2, 2)
    bh_returns = (results_df['BuyHold_Value'] / results_df['BuyHold_Value'].iloc[0] - 1) * 100
    rl_returns = (results_df['RL_Value'] / results_df['RL_Value'].iloc[0] - 1) * 100

    ax2.plot(results_df['Date'], bh_returns,
             label='Buy & Hold', linewidth=2, alpha=0.8)
    ax2.plot(results_df['Date'], rl_returns,
             label='RL Options Strategy', linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Stock Price
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(results_df['Date'], results_df['Stock_Price'],
             color='green', linewidth=2, alpha=0.7)
    ax3.set_title(f'{config.TICKER} Stock Price', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True, alpha=0.3)

    # 4. Strategy Outperformance
    ax4 = plt.subplot(2, 2, 4)
    outperformance = rl_returns - bh_returns
    colors = ['green' if x >= 0 else 'red' for x in outperformance]
    ax4.fill_between(results_df['Date'], 0, outperformance,
                      color='blue', alpha=0.3)
    ax4.plot(results_df['Date'], outperformance,
             color='blue', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('RL Strategy Outperformance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Outperformance (%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = results_dir / 'strategy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.close()

    # Create detailed performance metrics plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Load summary for metrics
    import json
    summary_path = results_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Extract metrics
        strategies = ['Buy & Hold', 'RL Strategy']
        sharpe_ratios = [
            float(summary['Buy and Hold']['Sharpe Ratio']),
            float(summary['RL Strategy']['Sharpe Ratio'])
        ]
        returns = [
            float(summary['Buy and Hold']['Total Return'].strip('%')),
            float(summary['RL Strategy']['Total Return'].strip('%'))
        ]

        # Sharpe Ratio comparison
        ax_sharpe = axes[0]
        bars1 = ax_sharpe.bar(strategies, sharpe_ratios, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax_sharpe.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target (1.0)')
        ax_sharpe.legend()
        ax_sharpe.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax_sharpe.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontweight='bold')

        # Total Return comparison
        ax_return = axes[1]
        bars2 = ax_return.bar(strategies, returns, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax_return.set_title('Total Return Comparison', fontsize=14, fontweight='bold')
        ax_return.set_ylabel('Return (%)')
        ax_return.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax_return.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}%',
                          ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        metrics_path = results_dir / 'metrics_comparison.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {metrics_path}")

    plt.close('all')
    print("\nAll visualizations created successfully!")


if __name__ == '__main__':
    create_visualizations()
