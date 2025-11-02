"""
Create visualizations of the optimal structured note performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = pd.read_csv('FINAL_optimal_note_results.csv')
results['start_date'] = pd.to_datetime(results['start_date'])

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

# Plot 1: Returns Distribution
ax1.hist(results['total_return'] * 100, bins=10, color='#2ecc71', alpha=0.7, edgecolor='black')
ax1.axvline(results['total_return'].mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {results["total_return"].mean()*100:.2f}%')
ax1.set_xlabel('3-Year Total Return (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('AI Energy Structured Note: Return Distribution (6 Rolling 3-Year Periods)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Returns Over Time
ax2.plot(results['start_date'], results['total_return'] * 100, marker='o', markersize=8, linewidth=2, color='#3498db')
ax2.fill_between(results['start_date'], 0, results['total_return'] * 100, alpha=0.3, color='#3498db')
ax2.axhline(results['total_return'].mean() * 100, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {results["total_return"].mean()*100:.2f}%')
ax2.set_xlabel('Period Start Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('3-Year Return (%)', fontsize=12, fontweight='bold')
ax2.set_title('Returns Across Different Market Periods', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Plot 3: Risk-Return Profile
annual_returns = (1 + results['total_return']) ** (1/3) - 1
mean_return = annual_returns.mean() * 100
std_return = annual_returns.std() * 100
sharpe = (annual_returns.mean() - 0.04) / annual_returns.std()

# Create scatter plot
ax3.scatter([std_return], [mean_return], s=300, color='#e74c3c', edgecolors='black', linewidths=2, zorder=5, label='Structured Note')

# Add Sharpe ratio isolines
x_range = np.linspace(0, std_return * 1.5, 100)
for target_sharpe in [0.5, 1.0, 1.5]:
    y_sharpe = target_sharpe * x_range + 4.0  # Risk-free rate = 4%
    ax3.plot(x_range, y_sharpe, linestyle='--', alpha=0.5, label=f'Sharpe = {target_sharpe:.1f}')

# Annotate the point
ax3.annotate(f'Sharpe = {sharpe:.3f}\nReturn = {mean_return:.2f}%\nVol = {std_return:.2f}%',
             xy=(std_return, mean_return),
             xytext=(std_return + 0.5, mean_return + 1),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))

ax3.set_xlabel('Annual Standard Deviation (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
ax3.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, std_return * 1.6)
ax3.set_ylim(0, mean_return * 1.3)

plt.tight_layout()
plt.savefig('structured_note_performance.png', dpi=300, bbox_inches='tight')
print("Visualization saved to structured_note_performance.png")

# Print summary statistics
print("\n" + "="*80)
print("FINAL PERFORMANCE SUMMARY")
print("="*80)
print(f"\n3-Year Returns:")
print(f"  Mean: {results['total_return'].mean()*100:.2f}%")
print(f"  Median: {results['total_return'].median()*100:.2f}%")
print(f"  Min: {results['total_return'].min()*100:.2f}%")
print(f"  Max: {results['total_return'].max()*100:.2f}%")
print(f"\nAnnualized Metrics:")
print(f"  Mean Return: {mean_return:.2f}%")
print(f"  Std Deviation: {std_return:.2f}%")
print(f"  Sharpe Ratio: {sharpe:.4f}")
print(f"\nStructure Details:")
print(f"  Autocall Rate: {results['autocalled'].sum() / len(results) * 100:.0f}%")
print(f"  Average Holding Period: {results['days_held'].mean():.0f} days")
print(f"  Average Coupons: ${results['coupons_received'].mean():,.2f}")
print("="*80)

plt.show()
