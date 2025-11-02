"""
AI Energy Transition Autocallable Structured Note
60% NVDA / 40% URA basket with autocall, buffer, and kicker features
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class AIEnergyStructuredNote:
    """
    Buffered Autocallable Structured Note on NVDA/URA basket

    Features:
    - 60/40 NVDA/URA basket
    - Quarterly autocall observations
    - Downside buffer protection
    - Step-up conditional coupons
    - Momentum kicker for enhanced upside
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        nvda_weight: float = 0.60,
        ura_weight: float = 0.40,
        autocall_threshold: float = 0.12,  # 12% above initial
        autocall_return: float = 0.07,  # 7% annual return on autocall
        buffer_level: float = 0.25,  # 25% downside buffer
        coupon_year1: float = 0.05,  # 5% annual
        coupon_year2: float = 0.07,  # 7% annual
        coupon_year3: float = 0.09,  # 9% annual
        kicker_start: float = 0.25,  # Kicker starts at +25%
        kicker_end: float = 0.60,  # Kicker ends at +60%
        kicker_multiplier: float = 1.75,  # 1.75x participation in kicker zone
        maturity_years: float = 3.0
    ):
        self.initial_capital = initial_capital
        self.nvda_weight = nvda_weight
        self.ura_weight = ura_weight
        self.autocall_threshold = autocall_threshold
        self.autocall_return = autocall_return
        self.buffer_level = buffer_level
        self.coupon_rates = {
            1: coupon_year1,
            2: coupon_year2,
            3: coupon_year3
        }
        self.kicker_start = kicker_start
        self.kicker_end = kicker_end
        self.kicker_multiplier = kicker_multiplier
        self.maturity_years = maturity_years
        self.maturity_days = int(maturity_years * 252)  # Trading days

    def create_basket(self, nvda_prices: pd.Series, ura_prices: pd.Series) -> pd.Series:
        """Create weighted basket of NVDA and URA"""
        # Normalize to start at 100
        nvda_normalized = nvda_prices / nvda_prices.iloc[0] * 100
        ura_normalized = ura_prices / ura_prices.iloc[0] * 100

        basket = (self.nvda_weight * nvda_normalized +
                 self.ura_weight * ura_normalized)

        return basket

    def calculate_basket_return(self, initial_value: float, final_value: float) -> float:
        """Calculate percentage return of basket"""
        return (final_value - initial_value) / initial_value

    def check_autocall(self, basket_value: float, initial_value: float,
                      days_held: int) -> Tuple[bool, float]:
        """
        Check if autocall is triggered
        Returns: (is_triggered, payout)
        """
        basket_return = self.calculate_basket_return(initial_value, basket_value)

        if basket_return >= self.autocall_threshold:
            years_held = days_held / 252
            payout = self.initial_capital * (1 + self.autocall_return * years_held)
            return True, payout

        return False, 0.0

    def calculate_coupon(self, basket_value: float, initial_value: float,
                        year: int, quarter: int) -> float:
        """
        Calculate conditional quarterly coupon
        Pays only if basket is positive at observation date
        """
        basket_return = self.calculate_basket_return(initial_value, basket_value)

        if basket_return >= 0:
            annual_rate = self.coupon_rates.get(year, self.coupon_rates[3])
            quarterly_coupon = self.initial_capital * (annual_rate / 4)
            return quarterly_coupon

        return 0.0

    def calculate_maturity_payoff(self, basket_return: float) -> float:
        """
        Calculate payoff at maturity with buffer and kicker

        Payoff structure:
        - If return > kicker_end (60%): 1:1 participation (capped)
        - If kicker_start < return <= kicker_end (25-60%): Enhanced participation
        - If buffer < return <= kicker_start: 1:1 participation
        - If -buffer <= return <= 0: Buffer protects, get 100% principal
        - If return < -buffer: Lose 1:1 below buffer
        """
        if basket_return < -self.buffer_level:
            # Below buffer: lose money
            # Loss = (return + buffer), e.g., -30% return with 25% buffer = -5% loss
            loss_beyond_buffer = basket_return + self.buffer_level
            payout = self.initial_capital * (1 + loss_beyond_buffer)

        elif -self.buffer_level <= basket_return < 0:
            # Within buffer: protected
            payout = self.initial_capital

        elif 0 <= basket_return < self.kicker_start:
            # Positive but below kicker: 1:1 participation
            payout = self.initial_capital * (1 + basket_return)

        elif self.kicker_start <= basket_return <= self.kicker_end:
            # Kicker zone: enhanced participation
            base_return = self.kicker_start
            kicker_return = (basket_return - self.kicker_start) * self.kicker_multiplier
            total_return = base_return + kicker_return
            payout = self.initial_capital * (1 + total_return)

        else:  # basket_return > kicker_end
            # Above kicker: revert to 1:1 (soft cap)
            base_return = self.kicker_start
            kicker_return = (self.kicker_end - self.kicker_start) * self.kicker_multiplier
            excess_return = (basket_return - self.kicker_end)
            total_return = base_return + kicker_return + excess_return
            payout = self.initial_capital * (1 + total_return)

        return payout

    def simulate_note(self, nvda_prices: pd.Series, ura_prices: pd.Series,
                     start_idx: int) -> Dict:
        """
        Simulate the structured note for a 3-year period starting at start_idx

        Returns dict with:
        - final_value: ending value
        - total_return: percentage return
        - autocalled: whether note was autocalled
        - autocall_date: date of autocall (if applicable)
        - coupons_received: total coupons collected
        - days_held: number of days held
        """
        # Extract 3-year window
        end_idx = min(start_idx + self.maturity_days, len(nvda_prices) - 1)
        nvda_window = nvda_prices.iloc[start_idx:end_idx+1]
        ura_window = ura_prices.iloc[start_idx:end_idx+1]

        # Create basket
        basket = self.create_basket(nvda_window, ura_window)
        initial_basket = basket.iloc[0]

        # Track coupons
        total_coupons = 0.0

        # Check for autocall quarterly (every 63 trading days ≈ 3 months)
        observation_dates = list(range(63, len(basket), 63))

        for obs_day in observation_dates:
            if obs_day >= len(basket):
                break

            basket_value = basket.iloc[obs_day]

            # Calculate which year and quarter we're in
            years_elapsed = obs_day / 252
            year = min(int(years_elapsed) + 1, 3)
            quarter = int((obs_day % 252) / 63) + 1

            # Check for coupon payment
            coupon = self.calculate_coupon(basket_value, initial_basket, year, quarter)
            total_coupons += coupon

            # Check for autocall
            is_autocalled, autocall_payout = self.check_autocall(
                basket_value, initial_basket, obs_day
            )

            if is_autocalled:
                final_value = autocall_payout + total_coupons
                total_return = (final_value - self.initial_capital) / self.initial_capital

                return {
                    'final_value': final_value,
                    'total_return': total_return,
                    'autocalled': True,
                    'autocall_date': basket.index[obs_day],
                    'coupons_received': total_coupons,
                    'days_held': obs_day,
                    'basket_return': self.calculate_basket_return(initial_basket, basket_value)
                }

        # If we reach maturity without autocall
        final_basket = basket.iloc[-1]
        basket_return = self.calculate_basket_return(initial_basket, final_basket)

        maturity_payout = self.calculate_maturity_payoff(basket_return)
        final_value = maturity_payout + total_coupons
        total_return = (final_value - self.initial_capital) / self.initial_capital

        return {
            'final_value': final_value,
            'total_return': total_return,
            'autocalled': False,
            'autocall_date': None,
            'coupons_received': total_coupons,
            'days_held': len(basket) - 1,
            'basket_return': basket_return
        }


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    # Annualize (assuming daily returns)
    annualized_return = mean_return * 252
    annualized_std = std_return * np.sqrt(252)

    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return sharpe


if __name__ == "__main__":
    # Quick test
    print("AI Energy Transition Structured Note - Test")
    print("="*60)

    # Load data
    nvda_data = pd.read_csv('nvda_data.csv', index_col=0, parse_dates=True)
    ura_data = pd.read_csv('ura_data.csv', index_col=0, parse_dates=True)

    # Create note
    note = AIEnergyStructuredNote()

    # Test simulation on first 3-year period
    result = note.simulate_note(nvda_data['Close'], ura_data['Close'], start_idx=0)

    print(f"\nTest Simulation Results:")
    print(f"Final Value: ${result['final_value']:,.2f}")
    print(f"Total Return: {result['total_return']*100:.2f}%")
    print(f"Autocalled: {result['autocalled']}")
    if result['autocalled']:
        print(f"Autocall Date: {result['autocall_date']}")
    print(f"Coupons Received: ${result['coupons_received']:,.2f}")
    print(f"Days Held: {result['days_held']}")
    print(f"Basket Return: {result['basket_return']*100:.2f}%")
