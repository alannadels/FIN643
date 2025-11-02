"""
Final optimization: Focus on structures that DON'T autocall early
Let the kicker zone capture the massive NVDA/URA gains
"""
import numpy as np
import pandas as pd
from structured_note import AIEnergyStructuredNote
from backtest_and_optimize import rolling_window_backtest
from maximize_sharpe import calculate_sharpe_from_results, print_final_results

def final_push_for_sharpe(nvda_prices, ura_prices):
    """
    Last optimization push - focus on letting gains run
    Key insight: Autocall is KILLING us in a bull market
    """

    configs = [
        {
            'name': 'No Autocall - Full Kicker Participation',
            'nvda_weight': 0.65,
            'autocall_threshold': 0.99,  # Effectively disabled
            'buffer_level': 0.20,
            'autocall_return': 0.15,
            'coupon_year1': 0.03,
            'kicker_start': 0.10,  # Low start
            'kicker_end': 1.50,  # Very wide zone
            'kicker_multiplier': 1.15  # Slight boost across entire range
        },
        {
            'name': 'Delayed Autocall at 50%',
            'nvda_weight': 0.60,
            'autocall_threshold': 0.50,  # Let it run to +50%
            'buffer_level': 0.15,
            'autocall_return': 0.20,  # If it hits, great return
            'coupon_year1': 0.02,
            'kicker_start': 0.15,
            'kicker_end': 0.80,
            'kicker_multiplier': 1.30
        },
        {
            'name': 'High Autocall Better Return',
            'nvda_weight': 0.60,
            'autocall_threshold': 0.40,
            'buffer_level': 0.20,
            'autocall_return': 0.18,  # Much better autocall return
            'coupon_year1': 0.03,
            'kicker_start': 0.20,
            'kicker_end': 0.70,
            'kicker_multiplier': 1.50
        },
        {
            'name': 'Lower Threshold Higher Return',
            'nvda_weight': 0.60,
            'autocall_threshold': 0.25,
            'buffer_level': 0.20,
            'autocall_return': 0.15,
            'coupon_year1': 0.04,
            'kicker_start': 0.20,
            'kicker_end': 0.70,
            'kicker_multiplier': 1.75
        },
        {
            'name': 'Mega Kicker No Autocall',
            'nvda_weight': 0.70,
            'autocall_threshold': 0.99,
            'buffer_level': 0.10,
            'autocall_return': 0.15,
            'coupon_year1': 0.00,  # No coupons
            'kicker_start': 0.05,  # Very low start
            'kicker_end': 2.00,  # Huge range
            'kicker_multiplier': 1.10  # Slight leverage everywhere
        },
        {
            'name': 'Balanced - 35% Autocall',
            'nvda_weight': 0.60,
            'autocall_threshold': 0.35,
            'buffer_level': 0.20,
            'autocall_return': 0.16,
            'coupon_year1': 0.03,
            'kicker_start': 0.20,
            'kicker_end': 0.70,
            'kicker_multiplier': 1.60
        },
        {
            'name': 'Wide Kicker Moderate Autocall',
            'nvda_weight': 0.65,
            'autocall_threshold': 0.30,
            'buffer_level': 0.15,
            'autocall_return': 0.14,
            'coupon_year1': 0.02,
            'kicker_start': 0.10,
            'kicker_end': 1.00,
            'kicker_multiplier': 1.25
        },
        {
            'name': 'Super Aggressive',
            'nvda_weight': 0.75,
            'autocall_threshold': 0.99,
            'buffer_level': 0.05,
            'autocall_return': 0.20,
            'coupon_year1': 0.00,
            'kicker_start': 0.00,  # Kicker from 0%
            'kicker_end': 3.00,
            'kicker_multiplier': 1.05  # 5% leverage across all gains
        }
    ]

    print("\n" + "="*80)
    print("FINAL SHARPE MAXIMIZATION - LETTING GAINS RUN")
    print("="*80)

    best_sharpe = -np.inf
    best_config = None
    best_results = None

    for config in configs:
        name = config.pop('name')
        print(f"\n{'-'*80}")
        print(f"Testing: {name}")
        print(f"{'-'*80}")

        note = AIEnergyStructuredNote(
            nvda_weight=config['nvda_weight'],
            ura_weight=1 - config['nvda_weight'],
            autocall_threshold=config['autocall_threshold'],
            autocall_return=config['autocall_return'],
            buffer_level=config['buffer_level'],
            coupon_year1=config['coupon_year1'],
            coupon_year2=config['coupon_year1'] + 0.02,
            coupon_year3=config['coupon_year1'] + 0.04,
            kicker_start=config['kicker_start'],
            kicker_end=config['kicker_end'],
            kicker_multiplier=config['kicker_multiplier']
        )

        results_df = rolling_window_backtest(note, nvda_prices, ura_prices, window_step=126)
        sharpe = calculate_sharpe_from_results(results_df)

        annual_returns = (1 + results_df['total_return']) ** (1/3) - 1

        print(f"Mean 3-year return: {results_df['total_return'].mean()*100:.2f}%")
        print(f"Mean annual return: {annual_returns.mean()*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Autocall rate: {results_df['autocalled'].sum() / len(results_df) * 100:.0f}%")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = config.copy()
            best_config['name'] = name
            best_results = results_df.copy()
            print(f"\n*** NEW BEST SHARPE: {sharpe:.4f} ***")

    return best_sharpe, best_config, best_results

def main():
    print("="*80)
    print("FINAL OPTIMIZATION RUN")
    print("="*80)

    nvda_data = pd.read_csv('nvda_data.csv', index_col=0, parse_dates=True)
    ura_data = pd.read_csv('ura_data.csv', index_col=0, parse_dates=True)

    nvda_prices = nvda_data['Close']
    ura_prices = ura_data['Close']

    best_sharpe, best_config, best_results = final_push_for_sharpe(nvda_prices, ura_prices)

    print_final_results(best_sharpe, best_config, best_results)

    # Save
    best_results.to_csv('FINAL_optimal_note_results.csv', index=False)

    import json
    with open('FINAL_optimal_parameters.json', 'w') as f:
        json.dump({
            'configuration_name': best_config.get('name', 'Optimal'),
            'sharpe_ratio': float(best_sharpe),
            'parameters': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                          for k, v in best_config.items() if k != 'name'}
        }, f, indent=2)

    print(f"\n\n🎯 **FINAL MAXIMUM SHARPE RATIO: {best_sharpe:.4f}** 🎯\n")

    return best_sharpe, best_config

if __name__ == "__main__":
    main()
