"""
Gym environment for options trading with RL
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import timedelta
import config
from utils import get_option_for_strategy, normalize_state


class OptionsTradeEnv(gym.Env):
    """
    Custom trading environment for options strategies

    Action Space:
        0: Buy and Hold (hold stock, no options)
        1: Covered Call (own stock + sell OTM call for income)
        2: Protective Put (own stock + buy OTM put for downside protection)

    State Space (8 features):
        - Normalized price relative to 200 SMA
        - RSI (normalized)
        - Volatility (normalized)
        - Trend (50 SMA vs 200 SMA)
        - Volume ratio
        - 5-day return
        - Distance from 52-week high
        - VIX proxy (volatility)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, stock_data, options_data=None, episode_length=60, initial_capital=100000):
        super().__init__()

        self.stock_data = stock_data
        self.options_data = options_data
        self.episode_length = episode_length
        self.initial_capital = initial_capital

        # Action space: 3 strategies
        # 0: Buy and Hold (no options)
        # 1: Covered Call (hold stock + sell call for income)
        # 2: Protective Put (hold stock + buy put for downside protection)
        self.action_space = spaces.Discrete(3)

        # State space: 8 normalized features
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(8,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.reserved_cash = 0  # Cash reserved for option obligations

        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.action_history = []

        # Current position
        self.current_action = 0
        self.option_position = None  # Current option if any
        self.option_entry_step = None  # When option was opened
        self.option_hold_days = 30  # Hold options for 30 days

        # Get valid date range (must have complete data)
        self.valid_dates = stock_data.dropna().index

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Randomly sample episode start date
        max_start_idx = len(self.valid_dates) - self.episode_length - 1
        if max_start_idx <= 0:
            # If episode length >= data length, start from beginning
            self.episode_start_idx = 0
        else:
            self.episode_start_idx = np.random.randint(0, max_start_idx)

        # Reset portfolio
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.shares = 0
        self.reserved_cash = 0
        self.current_action = 0
        self.option_position = None
        self.option_entry_step = None

        # Reset tracking
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        self.action_history = []

        # Get initial state
        state = self._get_state()

        return state, {}

    def _get_current_date(self):
        """Get current date in episode"""
        idx = self.episode_start_idx + self.current_step
        return self.valid_dates[idx]

    def _get_state(self):
        """Get current market state"""
        current_date = self._get_current_date()
        row = self.stock_data.loc[current_date]

        state_dict = {
            'price': row['Close'],
            'sma_50': row['SMA_50'],
            'sma_200': row['SMA_200'],
            'rsi': row['RSI'],
            'volatility': row['Volatility'],
            'volume_ratio': row['Volume_Ratio'],
            'return_5d': row['Return_5d'],
            'dist_from_high': row['Dist_from_High']
        }

        return normalize_state(state_dict, self.stock_data)

    def _get_stock_price(self):
        """Get current stock price"""
        current_date = self._get_current_date()
        return self.stock_data.loc[current_date, 'Close']

    def _execute_action(self, action):
        """Execute trading action"""
        stock_price = self._get_stock_price()
        current_date = self._get_current_date()

        # Check if we have an active option position that hasn't expired yet
        if self.option_position is not None and self.option_entry_step is not None:
            days_held = self.current_step - self.option_entry_step
            if days_held < self.option_hold_days:
                # Option still active, just maintain current stock position
                # Don't close or open new positions
                self.action_history.append(action)
                return
            else:
                # Option expired, settle it
                self._close_position()

        # No active option or option just expired - can execute new action
        if action == 0:
            # Buy and Hold - just hold stock
            if self.shares == 0:
                # Need to buy stock
                self._execute_buy_hold(stock_price)
            # else already holding, do nothing

        elif action == 1:
            # Covered Call - need stock + sell call
            if self.shares == 0:
                # First buy stock
                self._execute_buy_hold(stock_price)
            # Now sell call
            self._execute_covered_call(stock_price, current_date)

        elif action == 2:
            # Protective Put - need stock + buy put
            if self.shares == 0:
                # First buy stock
                self._execute_buy_hold(stock_price)
            # Now buy put
            self._execute_protective_put(stock_price, current_date)

        self.current_action = action
        self.action_history.append(action)

    def _close_position(self):
        """Close current position and settle to cash"""
        stock_price = self._get_stock_price()

        # Close option position if any
        if self.option_position is not None:
            self._settle_option()
            self.option_position = None
            self.option_entry_step = None

        # Sell all shares
        self.cash += self.shares * stock_price
        self.shares = 0

        # Release reserved cash
        self.reserved_cash = 0

    def _execute_buy_hold(self, stock_price):
        """Buy and hold stock"""
        shares_to_buy = int(self.cash / stock_price)
        cost = shares_to_buy * stock_price

        self.shares = shares_to_buy
        self.cash -= cost

    def _execute_covered_call(self, stock_price, current_date):
        """Covered call: own stock + sell call"""
        # Note: stock should already be owned at this point
        # Sell call option
        option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'covered_call'
        )

        if option is not None:
            # Collect premium (use bid price since we're selling)
            premium = option['bid'] * self.shares
            self.cash += premium

            self.option_position = {
                'type': 'short_call',
                'strike': option['strike'],
                'premium': premium,
                'dte': option['dte']
            }
            # Track when we opened this option
            self.option_entry_step = self.current_step

    def _execute_cash_put(self, stock_price, current_date):
        """Cash-secured put: sell put, hold cash"""
        option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'cash_put'
        )

        if option is not None:
            # Calculate number of contracts (1 contract = 100 shares)
            cash_for_assignment = option['strike'] * 100
            num_contracts = int(self.cash / cash_for_assignment)

            if num_contracts > 0:
                # Collect premium (use bid price since we're selling)
                premium = option['bid'] * num_contracts * 100
                self.cash += premium

                # Reserve cash for potential assignment
                self.reserved_cash = cash_for_assignment * num_contracts

                self.option_position = {
                    'type': 'short_put',
                    'strike': option['strike'],
                    'premium': premium,
                    'contracts': num_contracts,
                    'reserved_cash': self.reserved_cash,
                    'dte': option['dte']
                }
                # Track when we opened this option
                self.option_entry_step = self.current_step

    def _execute_protective_put(self, stock_price, current_date):
        """Protective put: own stock + buy put"""
        # Note: stock should already be owned at this point
        # Buy put option for protection
        option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'protective_put'
        )

        if option is not None:
            # Pay premium (use ask price since we're buying)
            premium = option['ask'] * self.shares
            self.cash -= premium

            self.option_position = {
                'type': 'long_put',
                'strike': option['strike'],
                'premium': premium,
                'dte': option['dte']
            }
            # Track when we opened this option
            self.option_entry_step = self.current_step

    def _execute_long_straddle(self, stock_price, current_date):
        """Long straddle: buy ATM call + ATM put"""
        # Get ATM call and put
        call_option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'straddle_call'
        )
        put_option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'straddle_put'
        )

        if call_option is not None and put_option is not None:
            # Calculate number of contracts based on available cash
            # Each straddle costs (call + put premium) * 100
            straddle_cost = (call_option['ask'] + put_option['ask']) * 100
            num_contracts = int(self.cash / straddle_cost)

            if num_contracts > 0:
                # Pay premiums for both options
                total_premium = straddle_cost * num_contracts
                self.cash -= total_premium

                self.option_position = {
                    'type': 'long_straddle',
                    'strike': call_option['strike'],  # ATM strike
                    'call_premium': call_option['ask'] * num_contracts * 100,
                    'put_premium': put_option['ask'] * num_contracts * 100,
                    'total_premium': total_premium,
                    'contracts': num_contracts,
                    'dte': call_option['dte']
                }
                # Track when we opened this option
                self.option_entry_step = self.current_step

    def _execute_collar(self, stock_price, current_date):
        """Collar: own stock + sell call + buy put"""
        # Buy stock
        shares_to_buy = int(self.cash / stock_price)
        cost = shares_to_buy * stock_price
        self.shares = shares_to_buy
        self.cash -= cost

        # Sell call
        call_option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'collar_call'
        )

        # Buy put
        put_option = get_option_for_strategy(
            self.options_data, current_date, stock_price, 'collar_put'
        )

        if call_option is not None and put_option is not None:
            # Net premium (collect from call, pay for put)
            call_premium = call_option['bid'] * self.shares
            put_premium = put_option['ask'] * self.shares
            net_premium = call_premium - put_premium

            self.cash += net_premium

            self.option_position = {
                'type': 'collar',
                'call_strike': call_option['strike'],
                'put_strike': put_option['strike'],
                'net_premium': net_premium,
                'dte': call_option['dte']
            }

    def _settle_option(self):
        """Settle option position at expiry"""
        if self.option_position is None:
            return

        stock_price = self._get_stock_price()
        option_type = self.option_position['type']

        if option_type == 'short_call':
            # Short call: lose money if stock > strike
            if stock_price > self.option_position['strike']:
                # Shares called away at strike
                proceeds = self.option_position['strike'] * self.shares
                self.cash += proceeds
                self.shares = 0

        elif option_type == 'short_put':
            # Short put: assigned if stock < strike
            if stock_price < self.option_position['strike']:
                # Buy shares at strike price
                num_shares = self.option_position['contracts'] * 100
                cost = self.option_position['strike'] * num_shares
                self.shares = num_shares
                self.cash -= cost

        elif option_type == 'long_put':
            # Long put: profit if stock < strike
            if stock_price < self.option_position['strike']:
                # Exercise put: sell at strike
                proceeds = self.option_position['strike'] * self.shares
                self.cash += proceeds
                self.shares = 0

        elif option_type == 'long_straddle':
            # Long straddle: profit from big moves either direction
            strike = self.option_position['strike']
            num_contracts = self.option_position['contracts']

            # Call side profit (if stock > strike)
            if stock_price > strike:
                call_profit = (stock_price - strike) * num_contracts * 100
                self.cash += call_profit

            # Put side profit (if stock < strike)
            if stock_price < strike:
                put_profit = (strike - stock_price) * num_contracts * 100
                self.cash += put_profit

            # Note: We already paid premiums upfront, so just collect intrinsic value

        elif option_type == 'collar':
            # Collar combines short call and long put
            call_strike = self.option_position['call_strike']
            put_strike = self.option_position['put_strike']

            if stock_price > call_strike:
                # Shares called away
                proceeds = call_strike * self.shares
                self.cash += proceeds
                self.shares = 0
            elif stock_price < put_strike:
                # Exercise put
                proceeds = put_strike * self.shares
                self.cash += proceeds
                self.shares = 0

    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        stock_price = self._get_stock_price()
        stock_value = self.shares * stock_price

        # Option value (mark-to-market would be complex, simplified here)
        option_value = 0
        if self.option_position is not None:
            # For simplicity, options expire worthless or at intrinsic value
            # This is a simplification - real implementation would mark-to-market
            pass

        # Portfolio value = available cash + stock value - reserved cash is still part of portfolio but not available
        # Actually, reserved cash is still part of portfolio, just not available for new positions
        return self.cash + stock_value + option_value

    def step(self, action):
        """Execute one time step"""
        # Execute action
        self._execute_action(action)

        # Move to next day
        self.current_step += 1

        # Calculate portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(self.portfolio_value)

        # Calculate return
        daily_return = (self.portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
        self.returns_history.append(daily_return)

        # Check if episode done
        done = self.current_step >= self.episode_length
        truncated = False

        # Calculate reward - optimize for beating buy-and-hold
        if len(self.returns_history) >= 5:
            # Calculate cumulative return so far
            portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

            # Calculate buy-and-hold return for same period
            initial_price = self.stock_data.iloc[self.episode_start_idx]['Close']
            current_price = self._get_stock_price()
            bh_return = (current_price - initial_price) / initial_price

            # Reward is outperformance vs buy-and-hold
            outperformance = (portfolio_return - bh_return) * 100  # Scale to percentage points

            # Also include absolute return component
            absolute_return_reward = portfolio_return * 50  # Scale up

            # Combined reward: favor both outperformance and absolute returns
            reward = outperformance * 2.0 + absolute_return_reward

            # Bonus for significantly beating buy-and-hold
            if portfolio_return > bh_return * 1.05:  # Beat by 5%+
                reward += 5.0

            # Penalize underperformance
            if portfolio_return < bh_return * 0.95:  # Underperform by 5%+
                reward -= 5.0

            # Penalize large drawdowns
            portfolio_array = np.array(self.portfolio_history)
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array[-1] - peak[-1]) / peak[-1]
            if drawdown < -0.25:  # More than 25% drawdown
                reward -= 2.0
        else:
            # Early in episode, reward positive returns
            reward = daily_return * 100  # Scale up for better learning

        # Get next state
        state = self._get_state() if not done else self._get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'return': daily_return,
            'action': action
        }

        return state, reward, done, truncated, info

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Shares: {self.shares}")
            print(f"Action: {self.current_action}")
