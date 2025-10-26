"""
Configuration file for RL Options Trading Strategy
"""

# Data parameters
TICKER = 'NCLH'
START_DATE = '2020-01-01'
END_DATE_TRAIN = '2024-12-31'
END_DATE_TEST = '2025-10-25'

# Train/validation split
VALIDATION_RATIO = 0.20  # 20% of training data for validation

# Episode parameters
EPISODE_LENGTH = 60  # Trading days per episode (2 months)
INITIAL_CAPITAL = 100000  # Starting with $100k

# RL hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95
TOTAL_TIMESTEPS = 100000

# Option parameters
OPTION_EXPIRY_DAYS = 30  # Look for ~30 day options
DELTA_RANGE_CALL = (0.25, 0.35)  # For covered calls - optimized for income generation
DELTA_RANGE_PUT = (-0.35, -0.25)  # For cash-secured puts
MIN_OPEN_INTEREST = 100  # Minimum OI for liquidity

# Technical indicators
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20
SMA_SHORT = 50
SMA_LONG = 200

# Risk parameters
MAX_DRAWDOWN_THRESHOLD = 0.30  # 30% max drawdown
RISK_FREE_RATE = 0.04  # Annual risk-free rate (4%)

# Model save path
MODEL_SAVE_PATH = 'trained_models/'
RESULTS_PATH = 'results/'
DATA_PATH = 'data/'
