"""
Train RL agent for options trading
"""

import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import config
from utils import load_data, split_data
from environment import OptionsTradeEnv


def make_env(stock_data, options_data):
    """Create environment factory"""
    def _init():
        return OptionsTradeEnv(
            stock_data=stock_data,
            options_data=options_data,
            episode_length=config.EPISODE_LENGTH,
            initial_capital=config.INITIAL_CAPITAL
        )
    return _init


def train_agent():
    """Train PPO agent on options trading"""
    print("=" * 80)
    print("Options Trading RL - Training")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    stock_data, options_data = load_data()

    # Split data
    train_stock, val_stock, test_stock, train_options, val_options, test_options = split_data(
        stock_data, options_data
    )

    # Create environments
    print("\nCreating training and validation environments...")
    train_env = DummyVecEnv([make_env(train_stock, train_options)])
    val_env = DummyVecEnv([make_env(val_stock, val_options)])

    # Create directories
    model_dir = Path(config.MODEL_SAVE_PATH)
    model_dir.mkdir(exist_ok=True)
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(model_dir / 'best_model'),
        log_path=str(results_dir),
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(model_dir / 'checkpoints'),
        name_prefix='ppo_options'
    )

    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        verbose=1,
        tensorboard_log=str(results_dir / 'tensorboard')
    )

    print("\nModel architecture:")
    print(f"  Policy: MLP")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  N steps: {config.N_STEPS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gamma: {config.GAMMA}")
    print(f"  GAE Lambda: {config.GAE_LAMBDA}")

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = model_dir / 'final_model'
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    return model


if __name__ == '__main__':
    train_agent()
