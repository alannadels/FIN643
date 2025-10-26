"""
Evaluate trained RL agent on test set and compare to buy-and-hold
"""

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
import config
from utils import load_data, split_data, calculate_sharpe_ratio, calculate_max_drawdown
from environment import OptionsTradeEnv


def evaluate_buy_and_hold(stock_data, initial_capital):
    """Evaluate simple buy-and-hold strategy"""
    initial_price = stock_data.iloc[0]['Close']
    final_price = stock_data.iloc[-1]['Close']

    shares = initial_capital / initial_price
    final_value = shares * final_price

    # Calculate returns
    portfolio_values = []
    for _, row in stock_data.iterrows():
        value = shares * row['Close']
        portfolio_values.append(value)

    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(portfolio_values)

    return {
        'strategy': 'Buy and Hold',
        'initial_value': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'portfolio_values': portfolio_values,
        'returns': returns
    }


def evaluate_rl_strategy(model, stock_data, options_data, initial_capital):
    """Evaluate RL agent strategy"""
    # Create environment with full test data
    env = OptionsTradeEnv(
        stock_data=stock_data,
        options_data=options_data,
        episode_length=len(stock_data) - 1,
        initial_capital=initial_capital
    )

    # Run episode
    obs, _ = env.reset()
    done = False
    portfolio_values = [initial_capital]
    actions_taken = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        portfolio_values.append(info['portfolio_value'])
        actions_taken.append(action)

        if truncated:
            break

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(portfolio_values)

    # Action distribution
    action_names = ['Buy & Hold', 'Covered Call', 'Protective Put']
    action_counts = {action_names[i]: np.sum(np.array(actions_taken) == i) for i in range(3)}

    return {
        'strategy': 'RL Options Strategy',
        'initial_value': initial_capital,
        'final_value': portfolio_values[-1],
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'actions': actions_taken,
        'action_distribution': action_counts
    }


def print_comparison(bh_results, rl_results):
    """Print comparison of strategies"""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Buy & Hold':>20} {'RL Strategy':>20} {'Difference':>15}")
    print("-" * 80)

    # Initial value
    print(f"{'Initial Capital':<25} ${bh_results['initial_value']:>19,.0f} "
          f"${rl_results['initial_value']:>19,.0f} {'-':>15}")

    # Final value
    print(f"{'Final Value':<25} ${bh_results['final_value']:>19,.0f} "
          f"${rl_results['final_value']:>19,.0f} "
          f"${rl_results['final_value'] - bh_results['final_value']:>14,.0f}")

    # Total return
    bh_return_pct = bh_results['total_return'] * 100
    rl_return_pct = rl_results['total_return'] * 100
    return_diff = rl_return_pct - bh_return_pct
    print(f"{'Total Return':<25} {bh_return_pct:>19.2f}% {rl_return_pct:>19.2f}% "
          f"{return_diff:>14.2f}%")

    # Sharpe ratio
    sharpe_diff = rl_results['sharpe_ratio'] - bh_results['sharpe_ratio']
    print(f"{'Sharpe Ratio':<25} {bh_results['sharpe_ratio']:>20.3f} "
          f"{rl_results['sharpe_ratio']:>20.3f} {sharpe_diff:>15.3f}")

    # Max drawdown
    bh_dd_pct = bh_results['max_drawdown'] * 100
    rl_dd_pct = rl_results['max_drawdown'] * 100
    dd_diff = rl_dd_pct - bh_dd_pct
    print(f"{'Max Drawdown':<25} {bh_dd_pct:>19.2f}% {rl_dd_pct:>19.2f}% "
          f"{dd_diff:>14.2f}%")

    # Action distribution
    if 'action_distribution' in rl_results:
        print("\n" + "=" * 80)
        print("RL STRATEGY - ACTION DISTRIBUTION")
        print("=" * 80)
        total_actions = sum(rl_results['action_distribution'].values())
        for action, count in rl_results['action_distribution'].items():
            pct = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"{action:<25} {count:>10} ({pct:>5.1f}%)")

    print("\n" + "=" * 80)


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("Options Trading RL - Evaluation on Test Set")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    stock_data, options_data = load_data()

    # Split data
    _, _, test_stock, _, _, test_options = split_data(stock_data, options_data)

    print(f"\nTest period: {test_stock.index[0]} to {test_stock.index[-1]}")
    print(f"Test days: {len(test_stock)}")

    # Load trained model
    model_path = Path(config.MODEL_SAVE_PATH) / 'best_model' / 'best_model.zip'
    if not model_path.exists():
        # Try final model
        model_path = Path(config.MODEL_SAVE_PATH) / 'final_model.zip'

    if not model_path.exists():
        print(f"\nError: No trained model found at {model_path}")
        print("Please run train.py first to train the model.")
        return

    print(f"\nLoading model from {model_path}")
    model = PPO.load(model_path)

    # Evaluate buy-and-hold
    print("\nEvaluating buy-and-hold strategy...")
    bh_results = evaluate_buy_and_hold(test_stock, config.INITIAL_CAPITAL)

    # Evaluate RL strategy
    print("Evaluating RL options strategy...")
    rl_results = evaluate_rl_strategy(model, test_stock, test_options, config.INITIAL_CAPITAL)

    # Print comparison
    print_comparison(bh_results, rl_results)

    # Save results
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(exist_ok=True)

    # Save to CSV
    results_df = pd.DataFrame({
        'Date': test_stock.index[:len(bh_results['portfolio_values'])],
        'BuyHold_Value': bh_results['portfolio_values'],
        'RL_Value': rl_results['portfolio_values'][:len(bh_results['portfolio_values'])],
        'Stock_Price': test_stock['Close'][:len(bh_results['portfolio_values'])]
    })
    results_df.to_csv(results_dir / 'test_results.csv', index=False)
    print(f"\nResults saved to {results_dir / 'test_results.csv'}")

    # Save summary
    summary = {
        'Buy and Hold': {
            'Total Return': f"{bh_results['total_return']*100:.2f}%",
            'Sharpe Ratio': f"{bh_results['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{bh_results['max_drawdown']*100:.2f}%",
            'Final Value': f"${bh_results['final_value']:,.0f}"
        },
        'RL Strategy': {
            'Total Return': f"{rl_results['total_return']*100:.2f}%",
            'Sharpe Ratio': f"{rl_results['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{rl_results['max_drawdown']*100:.2f}%",
            'Final Value': f"${rl_results['final_value']:,.0f}"
        }
    }

    import json
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return bh_results, rl_results


if __name__ == '__main__':
    main()
