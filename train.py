#!/usr/bin/env python3
"""
train.py — Train a PPO agent on the Stock Portfolio environment.

Usage:
    python scripts/train.py --timesteps 500000 --reward sharpe --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_env import StockPortfolioEnv
from agents.ppo_agent import train_ppo


def main():
    parser = argparse.ArgumentParser(description="Train PPO on StockPortfolio-v0")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--reward", type=str, default="log_return", choices=["log_return", "sharpe", "sortino"])
    parser.add_argument("--n-assets", type=int, default=5, help="Number of assets")
    parser.add_argument("--window", type=int, default=20, help="Observation window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Where to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Tensorboard log directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = StockPortfolioEnv(
        n_assets=args.n_assets,
        window=args.window,
        reward_mode=args.reward,
        seed=args.seed,
    )

    save_path = os.path.join(args.save_dir, f"ppo_{args.reward}_{args.timesteps}")

    print(f"{'='*60}")
    print(f"  Training PPO on StockPortfolio-v0")
    print(f"  Assets: {args.n_assets}  |  Window: {args.window}")
    print(f"  Reward: {args.reward}  |  Steps: {args.timesteps:,}")
    print(f"{'='*60}")

    model = train_ppo(
        env,
        total_timesteps=args.timesteps,
        seed=args.seed,
        save_path=save_path,
        log_dir=args.log_dir,
    )

    print(f"\nTraining complete. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
