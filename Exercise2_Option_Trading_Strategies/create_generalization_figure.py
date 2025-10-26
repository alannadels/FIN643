"""
Create visualization comparing RL strategy vs Buy-and-Hold across similar stocks
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from generalization test results
tickers = ['CCL', 'RCL', 'AAL', 'UAL', 'DAL']
bh_returns = [17.55, 39.48, -18.94, 4.07, 4.42]
rl_returns = [19.12, 20.83, -1.50, 11.14, 9.88]
outperformance = [1.57, -18.64, 17.45, 7.07, 5.45]

bh_sharpe = [0.574, 1.053, -0.243, 0.307, 0.284]
rl_sharpe = [0.610, 0.652, 0.176, 0.445, 0.402]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# --- Subplot 1: Total Returns Comparison ---
x = np.arange(len(tickers))
width = 0.35

bars1 = ax1.bar(x - width/2, bh_returns, width, label='Buy & Hold',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, rl_returns, width, label='RL Strategy',
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars - adjusted positioning to avoid overlap
for i, (bh, rl) in enumerate(zip(bh_returns, rl_returns)):
    # For positive values, place above bar; for negative, place below
    if bh > 0:
        ax1.text(i - width/2, bh + 1.5, f'{bh:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax1.text(i - width/2, bh - 1.5, f'{bh:.1f}%',
                 ha='center', va='top', fontsize=9, fontweight='bold')

    if rl > 0:
        ax1.text(i + width/2, rl + 1.5, f'{rl:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax1.text(i + width/2, rl - 1.5, f'{rl:.1f}%',
                 ha='center', va='top', fontsize=9, fontweight='bold')

ax1.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
ax1.set_title('NCLH Model Generalization: Total Returns on Similar Stocks (2025)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(tickers, fontsize=11)
ax1.legend(fontsize=11, loc='upper left')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# --- Subplot 2: Sharpe Ratio Comparison ---
bars3 = ax2.bar(x - width/2, bh_sharpe, width, label='Buy & Hold',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars4 = ax2.bar(x + width/2, rl_sharpe, width, label='RL Strategy',
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bh, rl) in enumerate(zip(bh_sharpe, rl_sharpe)):
    ax2.text(i - width/2, bh + (0.04 if bh > 0 else -0.06), f'{bh:.3f}',
             ha='center', va='bottom' if bh > 0 else 'top', fontsize=9, fontweight='bold')
    ax2.text(i + width/2, rl + (0.04 if rl > 0 else -0.06), f'{rl:.3f}',
             ha='center', va='bottom' if rl > 0 else 'top', fontsize=9, fontweight='bold')

ax2.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_title('NCLH Model Generalization: Sharpe Ratios on Similar Stocks (2025)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(tickers, fontsize=11)
ax2.legend(fontsize=11, loc='upper left')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target Sharpe (1.0)')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Removed summary statistics box

plt.tight_layout()
plt.savefig('results/generalization_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved to results/generalization_comparison.png")
plt.show()
