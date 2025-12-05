"""
Main script for Options Strategy
Run with: python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import config
from data_download import get_data
from signals import generate_all_signals
from strategy import generate_strategy_decisions, summarize_decisions
from backtest import (Backtester, generate_performance_report, save_results)


def main():
    print("=" * 70)
    print("       OPTIONS STRATEGY - FINAL EXAM")
    print("=" * 70)
    print(f"\nAssets: {config.EQUITY_TICKER} (Equity) + {config.FIXED_INCOME_TICKER} (Fixed Income)")
    print(f"Period: {config.START_DATE} to {config.END_DATE}")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,}")

    # =========================================================================
    # STEP 1: Download/Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    equity_data = get_data(config.EQUITY_TICKER, config.START_DATE, config.END_DATE,
                           config.DATA_PATH)
    fi_data = get_data(config.FIXED_INCOME_TICKER, config.START_DATE, config.END_DATE,
                       config.DATA_PATH)

    # =========================================================================
    # STEP 2: Generate Trading Signals
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING TRADING SIGNALS")
    print("=" * 70)

    signals = generate_all_signals(equity_data, fi_data)

    print(f"\nSignal Statistics:")
    print(f"  Equity RSI Range: {signals['Equity_RSI'].min():.1f} - {signals['Equity_RSI'].max():.1f}")
    print(f"  Equity Volatility Range: {signals['Equity_Volatility'].min()*100:.1f}% - {signals['Equity_Volatility'].max()*100:.1f}%")
    print(f"  Equity IV Rank Range: {signals['Equity_IV_Rank'].min():.1f} - {signals['Equity_IV_Rank'].max():.1f}")

    # =========================================================================
    # STEP 3: Generate Strategy Decisions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING STRATEGY DECISIONS")
    print("=" * 70)

    decisions = generate_strategy_decisions(signals)
    print(summarize_decisions(decisions))

    # =========================================================================
    # STEP 4: Run Backtest
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: RUNNING BACKTEST")
    print("=" * 70)

    backtester = Backtester(equity_data, fi_data, signals, decisions,
                            config.INITIAL_CAPITAL)

    print("\nRunning strategies...")
    strategy_results = backtester.run_strategy(config.REBALANCE_FREQUENCY)
    equity_results = backtester.run_buy_and_hold_equity()
    fi_results = backtester.run_buy_and_hold_fi()
    equal_results = backtester.run_equal_weight()

    print(f"  Strategy backtest: {len(strategy_results)} days")
    print(f"  Benchmarks: {len(equity_results)} days")

    # =========================================================================
    # STEP 5: Calculate Metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CALCULATING METRICS")
    print("=" * 70)

    strategy_metrics = backtester.calculate_metrics(strategy_results)
    equity_metrics = backtester.calculate_metrics(equity_results)
    fi_metrics = backtester.calculate_metrics(fi_results)
    equal_metrics = backtester.calculate_metrics(equal_results)

    # =========================================================================
    # STEP 6: Generate Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING REPORT")
    print("=" * 70)

    report = generate_performance_report(strategy_metrics, equity_metrics,
                                         fi_metrics, equal_metrics, decisions)

    print(report)

    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)

    save_results(report, strategy_results, equity_results, fi_results, equal_results)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
