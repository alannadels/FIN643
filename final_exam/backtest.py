"""
Backtesting Engine for Options Strategy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from option_pricing import black_scholes_call, black_scholes_put
import config


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = None) -> float:
    """Calculate annualized Sharpe ratio"""
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)


def calculate_total_return(initial: float, final: float) -> float:
    """Calculate total return as percentage"""
    return (final - initial) / initial


def calculate_strategy_return(strategy: int, entry_price: float, exit_price: float,
                              volatility: float, days_held: int) -> float:
    """
    Calculate return multiplier from option strategy

    Returns: multiplier where 1.0 = no change
    """
    if volatility <= 0 or np.isnan(volatility):
        volatility = 0.25

    T = max(days_held / 252, 0.01)
    r = config.RISK_FREE_RATE
    stock_return = (exit_price - entry_price) / entry_price

    if strategy == 0:  # Covered Call
        # Sell OTM call, collect premium
        strike = entry_price * 1.03  # 3% OTM for good premium
        premium = black_scholes_call(entry_price, strike, T, r, volatility)
        premium_pct = premium / entry_price * 0.95  # Bid price

        if exit_price > strike:
            # Called away - gain capped at strike + premium
            return 1 + (strike - entry_price) / entry_price + premium_pct
        else:
            # Keep stock + premium
            return 1 + stock_return + premium_pct

    elif strategy == 1:  # Protective Put
        # Buy OTM put for protection
        strike = entry_price * 0.95  # 5% OTM
        premium = black_scholes_put(entry_price, strike, T, r, volatility)
        premium_pct = premium / entry_price * 1.05  # Ask price

        if exit_price < strike:
            # Put exercised - loss limited
            return 1 + (strike - entry_price) / entry_price - premium_pct
        else:
            # Stock return minus premium cost
            return 1 + stock_return - premium_pct

    elif strategy == 2:  # Long Straddle
        # Buy ATM call + ATM put
        strike = entry_price
        call_prem = black_scholes_call(entry_price, strike, T, r, volatility)
        put_prem = black_scholes_put(entry_price, strike, T, r, volatility)
        total_prem_pct = (call_prem + put_prem) / entry_price * 1.05  # Ask

        call_payoff = max(exit_price - strike, 0) / entry_price
        put_payoff = max(strike - exit_price, 0) / entry_price

        return 1 + call_payoff + put_payoff - total_prem_pct

    elif strategy == 3:  # Long Strangle
        # Buy OTM call + OTM put
        call_strike = entry_price * 1.05
        put_strike = entry_price * 0.95
        call_prem = black_scholes_call(entry_price, call_strike, T, r, volatility)
        put_prem = black_scholes_put(entry_price, put_strike, T, r, volatility)
        total_prem_pct = (call_prem + put_prem) / entry_price * 1.05

        call_payoff = max(exit_price - call_strike, 0) / entry_price
        put_payoff = max(put_strike - exit_price, 0) / entry_price

        return 1 + call_payoff + put_payoff - total_prem_pct

    # Fallback: simple stock return
    return 1 + stock_return


class Backtester:
    """Backtesting engine for the implemented strategy"""

    def __init__(self, equity_data: pd.DataFrame, fi_data: pd.DataFrame,
                 signals: pd.DataFrame, decisions: pd.DataFrame,
                 initial_capital: float = 100000):

        self.initial_capital = initial_capital

        # Align all data
        common_dates = signals.index.intersection(equity_data.index).intersection(fi_data.index)
        common_dates = common_dates.intersection(decisions.index)

        self.dates = common_dates
        self.equity = equity_data.loc[common_dates]
        self.fi = fi_data.loc[common_dates]
        self.signals = signals.loc[common_dates]
        self.decisions = decisions.loc[common_dates]

    def run_strategy(self, rebalance_freq: int = 21) -> pd.DataFrame:
        """Run the implemented strategy backtest"""
        results = []
        portfolio_value = self.initial_capital

        rebalance_dates = list(range(0, len(self.dates), rebalance_freq))

        for period_idx, start_idx in enumerate(rebalance_dates):
            # End of period
            if period_idx + 1 < len(rebalance_dates):
                end_idx = rebalance_dates[period_idx + 1] - 1
            else:
                end_idx = len(self.dates) - 1

            start_date = self.dates[start_idx]
            end_date = self.dates[end_idx]

            # Get strategy decision at start of period
            decision = self.decisions.loc[start_date]
            strategy = int(decision['strategy'])
            equity_weight = decision['equity_weight']
            fi_weight = decision['fi_weight']

            # Get prices
            equity_start = self.equity.loc[start_date, 'Close']
            equity_end = self.equity.loc[end_date, 'Close']
            fi_start = self.fi.loc[start_date, 'Close']
            fi_end = self.fi.loc[end_date, 'Close']

            # Get volatility from signals
            vol = self.signals.loc[start_date, 'Equity_Volatility']
            days_held = end_idx - start_idx + 1

            # Calculate returns for the full period
            equity_option_mult = calculate_strategy_return(strategy, equity_start, equity_end, vol, days_held)
            fi_mult = fi_end / fi_start

            # Combined portfolio return for full period
            period_mult = equity_weight * equity_option_mult + fi_weight * fi_mult

            # Apply transaction cost
            period_mult *= (1 - config.TRANSACTION_COST)

            # Portfolio value at START of this period
            portfolio_value_start = portfolio_value

            # Record daily values for this period
            for day_idx in range(start_idx, end_idx + 1):
                date = self.dates[day_idx]

                if day_idx == start_idx:
                    # First day of period: value equals starting value
                    day_value = portfolio_value_start
                else:
                    day_equity = self.equity.iloc[day_idx]['Close']
                    day_fi = self.fi.iloc[day_idx]['Close']
                    days_so_far = day_idx - start_idx

                    # Interpolate daily value based on price movement
                    day_equity_mult = calculate_strategy_return(
                        strategy, equity_start, day_equity, vol, days_so_far
                    )
                    day_fi_mult = day_fi / fi_start
                    day_mult = equity_weight * day_equity_mult + fi_weight * day_fi_mult

                    day_value = portfolio_value_start * day_mult

                results.append({
                    'date': date,
                    'portfolio_value': day_value,
                    'strategy': strategy,
                    'equity_weight': equity_weight
                })

            # Update portfolio value for next period
            portfolio_value = portfolio_value * period_mult

        return pd.DataFrame(results).set_index('date')

    def run_buy_and_hold_equity(self) -> pd.DataFrame:
        """Buy and hold equity only"""
        results = []
        initial_price = self.equity.iloc[0]['Close']
        shares = self.initial_capital / initial_price

        for date in self.dates:
            price = self.equity.loc[date, 'Close']
            results.append({'date': date, 'portfolio_value': shares * price})

        return pd.DataFrame(results).set_index('date')

    def run_buy_and_hold_fi(self) -> pd.DataFrame:
        """Buy and hold fixed income only"""
        results = []
        initial_price = self.fi.iloc[0]['Close']
        shares = self.initial_capital / initial_price

        for date in self.dates:
            price = self.fi.loc[date, 'Close']
            results.append({'date': date, 'portfolio_value': shares * price})

        return pd.DataFrame(results).set_index('date')

    def run_equal_weight(self) -> pd.DataFrame:
        """50/50 equity/fixed income without options"""
        results = []
        equity_shares = (self.initial_capital * 0.5) / self.equity.iloc[0]['Close']
        fi_shares = (self.initial_capital * 0.5) / self.fi.iloc[0]['Close']

        for date in self.dates:
            equity_val = equity_shares * self.equity.loc[date, 'Close']
            fi_val = fi_shares * self.fi.loc[date, 'Close']
            results.append({'date': date, 'portfolio_value': equity_val + fi_val})

        return pd.DataFrame(results).set_index('date')

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """Calculate performance metrics"""
        values = results['portfolio_value'].values
        returns = np.diff(values) / values[:-1]

        start = results.index[0]
        end = results.index[-1]
        years = (end - start).days / 365.25

        return {
            'total_return': calculate_total_return(values[0], values[-1]),
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(values),
            'cagr': (values[-1] / values[0]) ** (1/years) - 1 if years > 0 else 0,
            'volatility': np.std(returns) * np.sqrt(252),
            'final_value': values[-1],
            'initial_value': values[0]
        }


def generate_performance_report(strategy_metrics: dict, equity_metrics: dict,
                                fi_metrics: dict, equal_metrics: dict,
                                decisions: pd.DataFrame) -> str:
    """Generate performance summary report"""

    # Strategy breakdown
    strategy_counts = decisions['strategy'].value_counts()
    total_periods = len(decisions)
    avg_equity_weight = decisions['equity_weight'].mean()

    equity_ticker = config.EQUITY_TICKER
    fi_ticker = config.FIXED_INCOME_TICKER

    report = """
================================================================================
               IMPLEMENTED OPTIONS STRATEGY - PERFORMANCE REPORT
================================================================================

Initial Capital: ${:,.0f}
Period: {} to {}

--------------------------------------------------------------------------------
                              PERFORMANCE METRICS
--------------------------------------------------------------------------------
Strategy                    | Total Return | Sharpe | Max Drawdown | Final Value
--------------------------------------------------------------------------------
Implemented Strategy        |   {:+7.1f}%    |  {:5.2f} |    {:6.1f}%   |  ${:,.0f}
{} Buy-and-Hold             |   {:+7.1f}%    |  {:5.2f} |    {:6.1f}%   |  ${:,.0f}
{} Buy-and-Hold             |   {:+7.1f}%    |  {:5.2f} |    {:6.1f}%   |  ${:,.0f}
50/50 {}/{} (No Options)  |   {:+7.1f}%    |  {:5.2f} |    {:6.1f}%   |  ${:,.0f}
--------------------------------------------------------------------------------

Outperformance vs {}:  {:+.1f}%
Outperformance vs {}: {:+.1f}%

--------------------------------------------------------------------------------
                           STRATEGY SELECTION BREAKDOWN
--------------------------------------------------------------------------------
""".format(
        config.INITIAL_CAPITAL,
        config.START_DATE, config.END_DATE,
        strategy_metrics['total_return'] * 100,
        strategy_metrics['sharpe_ratio'],
        strategy_metrics['max_drawdown'] * 100,
        strategy_metrics['final_value'],
        equity_ticker,
        equity_metrics['total_return'] * 100,
        equity_metrics['sharpe_ratio'],
        equity_metrics['max_drawdown'] * 100,
        equity_metrics['final_value'],
        fi_ticker,
        fi_metrics['total_return'] * 100,
        fi_metrics['sharpe_ratio'],
        fi_metrics['max_drawdown'] * 100,
        fi_metrics['final_value'],
        equity_ticker, fi_ticker,
        equal_metrics['total_return'] * 100,
        equal_metrics['sharpe_ratio'],
        equal_metrics['max_drawdown'] * 100,
        equal_metrics['final_value'],
        equity_ticker,
        (strategy_metrics['total_return'] - equity_metrics['total_return']) * 100,
        fi_ticker,
        (strategy_metrics['total_return'] - fi_metrics['total_return']) * 100
    )

    for strat_id, name in config.STRATEGIES.items():
        count = strategy_counts.get(strat_id, 0)
        pct = count / total_periods * 100 if total_periods > 0 else 0
        report += f"{name}:    {pct:.0f}% of periods\n"

    report += """
Average {} Allocation: {:.0f}%
Average {} Allocation: {:.0f}%

--------------------------------------------------------------------------------
                              KEY INSIGHTS
--------------------------------------------------------------------------------
- Strategy uses trading signals (RSI, Trend, Volatility, IV Rank) to select
  among 4 option strategies: Covered Call, Protective Put, Straddle, Strangle
- Allocation between {} and {} adjusts based on momentum and risk signals
- Covered calls generate income in flat/bullish markets
- Protective puts limit downside in volatile downtrends
- Straddles/strangles profit from large price moves when IV is low

================================================================================
""".format(equity_ticker, avg_equity_weight * 100,
           fi_ticker, (1 - avg_equity_weight) * 100,
           equity_ticker, fi_ticker)

    return report


def save_results(report: str, strategy_results: pd.DataFrame,
                 equity_results: pd.DataFrame, fi_results: pd.DataFrame,
                 equal_results: pd.DataFrame) -> None:
    """Save all results to files"""
    path = Path(config.RESULTS_PATH)
    path.mkdir(parents=True, exist_ok=True)

    # Save report
    with open(path / 'performance_summary.txt', 'w') as f:
        f.write(report)
    print(f"Report saved to {path / 'performance_summary.txt'}")

    # Save portfolio values
    equity_ticker = config.EQUITY_TICKER
    fi_ticker = config.FIXED_INCOME_TICKER
    all_results = pd.DataFrame({
        'Strategy': strategy_results['portfolio_value'],
        f'{equity_ticker}_BuyHold': equity_results['portfolio_value'],
        f'{fi_ticker}_BuyHold': fi_results['portfolio_value'],
        'Equal_Weight': equal_results['portfolio_value']
    })
    all_results.to_csv(path / 'portfolio_values.csv')
    print(f"Portfolio values saved to {path / 'portfolio_values.csv'}")
