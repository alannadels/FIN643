# Exercise 2: Options Trading Strategy with Reinforcement Learning

This project implements a reinforcement learning agent to trade NVIDIA (NVDA) options, aiming to beat buy-and-hold returns while maintaining a Sharpe ratio ≥ 1.0.

## Strategy Overview

The RL agent can choose from 5 different strategies each trading day:
1. **Buy and Hold**: Simply hold the stock
2. **Covered Call**: Own stock + sell OTM call (generate income)
3. **Cash-Secured Put**: Sell OTM put (collect premium, potentially buy stock at discount)
4. **Protective Put**: Own stock + buy OTM put (downside protection)
5. **Collar**: Own stock + sell call + buy put (capped upside and downside)

The agent learns which strategy to use based on market conditions (RSI, volatility, trend, etc.) to maximize risk-adjusted returns.

## Project Structure

```
Exercise2_Option_Trading_Strategies/
├── config.py              # Configuration and hyperparameters
├── requirements.txt       # Python dependencies
├── data_download.py       # Download stock and options data
├── black_scholes.py       # Black-Scholes option pricing (fallback)
├── utils.py               # Helper functions
├── environment.py         # Gym trading environment
├── train.py               # Train RL agent
├── evaluate.py            # Evaluate on test set
├── visualize.py           # Create comparison plots
├── main.py                # Run full pipeline
├── data/                  # Downloaded data
├── trained_models/        # Saved models
└── results/               # Results and visualizations
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data:
```bash
python data_download.py
```

This will:
- Download 5 years of NVDA stock data (2020-2024)
- Attempt to download historical options from WRDS (if you have access)
- Fall back to Black-Scholes simulated options if WRDS unavailable

## Usage

### Run Full Pipeline (Recommended)

Train the agent, evaluate on 2025 test set, and create visualizations:
```bash
python main.py
```

This will:
1. Train the PPO agent on 2020-2024 data (80% train, 20% validation)
2. Evaluate on 2025 test period
3. Generate comparison plots
4. Save results to `results/` directory

### Run Individual Steps

Train only:
```bash
python main.py --train-only
```

Evaluate only (requires trained model):
```bash
python main.py --eval-only
```

Create visualizations only:
```bash
python main.py --viz-only
```

## Configuration

Edit `config.py` to change:
- Training hyperparameters (learning rate, batch size, etc.)
- Episode length
- Option selection criteria (delta ranges, expiry)
- Risk parameters (max drawdown threshold, risk-free rate)

## Data Split

To avoid regime bias, the data is split as follows:
- **Training**: 80% of 2020-2024 data (random episodes)
- **Validation**: 20% of 2020-2024 data (random sample)
- **Test**: All of 2025 data (held out completely)

Random episode sampling ensures the agent sees diverse market conditions (bull/bear, high/low vol).

## Results

After running the pipeline, check:
- `results/strategy_comparison.png` - Portfolio value and returns over time
- `results/metrics_comparison.png` - Sharpe ratio and total return comparison
- `results/summary.json` - Performance metrics (Sharpe, returns, drawdown)
- `results/test_results.csv` - Daily portfolio values for both strategies

## Technical Details

### State Space (8 features)
- Price relative to 200-day SMA
- RSI (normalized)
- Volatility (normalized)
- Trend (50 SMA vs 200 SMA)
- Volume ratio
- 5-day return
- Distance from 52-week high
- VIX proxy

### Reward Function
The agent is rewarded based on:
- Sharpe ratio of recent returns (risk-adjusted performance)
- Penalty for large drawdowns (>20%)

### Algorithm
- **PPO (Proximal Policy Optimization)** from Stable Baselines3
- MLP policy with default architecture
- Training: 50,000 timesteps
- Episode length: 60 trading days (2 months)

## Requirements

- Python 3.8+
- See requirements.txt for full list
- Optional: WRDS access for historical options data

## Notes

- If WRDS data is unavailable, the system automatically generates simulated options using Black-Scholes
- Options are selected based on delta (0.25-0.35 for calls, -0.35 to -0.25 for puts)
- All options have ~30 days to expiry
- Transaction costs are simplified for this academic exercise
