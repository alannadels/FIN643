"""
Visualization module for Options Strategy
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import config


def create_performance_table_figure():
    """Create a professional figure of the performance metrics table"""

    # Performance data
    data = {
        'Strategy': ['Implemented Strategy', 'CVX Buy-and-Hold', 'HYG Buy-and-Hold', '50/50 CVX/HYG (No Options)'],
        'Total Return': ['+68.0%', '+57.4%', '+11.1%', '+34.3%'],
        'Sharpe': ['0.85', '0.53', '0.14', '0.44'],
        'Max Drawdown': ['-9.7%', '-24.9%', '-15.8%', '-17.6%'],
        'Final Value': ['$164,113', '$157,414', '$111,128', '$134,271']
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=[data['Strategy'], data['Total Return'], data['Sharpe'],
                  data['Max Drawdown'], data['Final Value']],
        rowLabels=['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Final Value'],
        colLabels=None,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Color the header row and first column
    for i in range(5):
        table[(i, 0)].set_facecolor('#4472C4')
        table[(i, 0)].set_text_props(weight='bold', color='white')

    # Highlight the implemented strategy column
    for i in range(5):
        table[(i, 0)].set_facecolor('#4472C4')

    # Alternate row colors for readability
    for i in range(5):
        for j in range(1, 4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    path = Path(config.RESULTS_PATH)
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'performance_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved performance table to {path / 'performance_table.png'}")

    plt.close()


def create_performance_bar_chart():
    """Create a bar chart comparing key metrics"""

    strategies = ['Implemented\nStrategy', 'CVX\nBuy-and-Hold', 'HYG\nBuy-and-Hold', '50/50\n(No Options)']

    # Data
    returns = [68.0, 57.4, 11.1, 34.3]
    sharpe = [0.85, 0.53, 0.14, 0.44]
    max_dd = [9.7, 24.9, 15.8, 17.6]  # Positive for visualization

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = ['#2E7D32', '#1976D2', '#F57C00', '#7B1FA2']

    # Total Return
    ax1 = axes[0]
    bars1 = ax1.bar(strategies, returns, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Total Return (%)', fontsize=12)
    ax1.set_title('Total Return', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars1, returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'+{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, max(returns) * 1.15)

    # Sharpe Ratio
    ax2 = axes[1]
    bars2 = ax2.bar(strategies, sharpe, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars2, sharpe):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(sharpe) * 1.2)

    # Max Drawdown (shown as negative)
    ax3 = axes[2]
    bars3 = ax3.bar(strategies, [-x for x in max_dd], color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Max Drawdown (%)', fontsize=12)
    ax3.set_title('Max Drawdown', fontsize=13, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars3, max_dd):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                f'-{val:.1f}%', ha='center', va='top', fontsize=10, fontweight='bold')
    ax3.set_ylim(min([-x for x in max_dd]) * 1.2, 0)

    plt.suptitle('Strategy Performance Comparison (2021-2024)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    path = Path(config.RESULTS_PATH)
    plt.savefig(path / 'performance_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved performance comparison to {path / 'performance_comparison.png'}")

    plt.close()


def create_portfolio_value_chart():
    """Create a line chart showing portfolio value over time"""

    # Load portfolio values
    path = Path(config.RESULTS_PATH)
    df = pd.read_csv(path / 'portfolio_values.csv', index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each strategy
    ax.plot(df.index, df['Strategy'], label='Implemented Strategy',
            color='#2E7D32', linewidth=2.5)
    ax.plot(df.index, df[f'{config.EQUITY_TICKER}_BuyHold'],
            label=f'{config.EQUITY_TICKER} Buy-and-Hold', color='#1976D2', linewidth=1.5, alpha=0.8)
    ax.plot(df.index, df[f'{config.FIXED_INCOME_TICKER}_BuyHold'],
            label=f'{config.FIXED_INCOME_TICKER} Buy-and-Hold', color='#F57C00', linewidth=1.5, alpha=0.8)
    ax.plot(df.index, df['Equal_Weight'], label='50/50 (No Options)',
            color='#7B1FA2', linewidth=1.5, alpha=0.8)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Portfolio Value Over Time (2021-2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add horizontal line at initial capital
    ax.axhline(y=100000, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(df.index[0], 102000, 'Initial Capital: $100,000', fontsize=9, color='gray')

    plt.tight_layout()

    # Save figure
    plt.savefig(path / 'portfolio_value.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved portfolio value chart to {path / 'portfolio_value.png'}")

    plt.close()


if __name__ == '__main__':
    print("Generating visualizations...")
    create_performance_bar_chart()
    create_portfolio_value_chart()
    print("\nAll visualizations saved to results/ folder")
