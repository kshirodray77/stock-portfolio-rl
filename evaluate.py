#!/usr/bin/env python3
"""
evaluate.py — Benchmark RL agent vs baselines on the portfolio env.

Usage:
    python scripts/evaluate.py --episodes 50 --seed 123
    python scripts/evaluate.py --model checkpoints/ppo_sharpe_200000 --episodes 50
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_env import StockPortfolioEnv
from agents.baselines import (
    UniformAgent,
    CashOnlyAgent,
    MomentumAgent,
    MeanVarianceAgent,
    RandomAgent,
)


def evaluate_agent(env, agent, n_episodes: int = 20, name: str = "Agent"):
    """Run n_episodes and return summary statistics."""
    returns = []
    sharpe_ratios = []
    max_drawdowns = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 1000)
        total_reward = 0
        ep_returns = []
        peak = env.initial_cash
        max_dd = 0

        done = False
        while not done:
            action = agent.act(obs) if hasattr(agent, "act") else agent.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_returns.append(reward)

            val = info["portfolio_value"]
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

            done = terminated or truncated

        final_return = info["total_return"]
        returns.append(final_return)
        max_drawdowns.append(max_dd)

        if len(ep_returns) > 1:
            arr = np.array(ep_returns)
            std = arr.std()
            sharpe = arr.mean() / std if std > 1e-8 else 0.0
            sharpe_ratios.append(sharpe * np.sqrt(252))

    results = {
        "name": name,
        "mean_return": np.mean(returns) * 100,
        "std_return": np.std(returns) * 100,
        "median_return": np.median(returns) * 100,
        "mean_sharpe": np.mean(sharpe_ratios),
        "mean_max_dd": np.mean(max_drawdowns) * 100,
        "best_return": np.max(returns) * 100,
        "worst_return": np.min(returns) * 100,
    }
    return results


def print_results(all_results: list[dict]):
    header = f"{'Agent':<20} {'Mean %':>8} {'Std %':>8} {'Sharpe':>8} {'MaxDD %':>8} {'Best %':>8} {'Worst %':>8}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for r in all_results:
        print(
            f"{r['name']:<20} "
            f"{r['mean_return']:>+7.2f}% "
            f"{r['std_return']:>7.2f}% "
            f"{r['mean_sharpe']:>8.3f} "
            f"{r['mean_max_dd']:>7.2f}% "
            f"{r['best_return']:>+7.2f}% "
            f"{r['worst_return']:>+7.2f}%"
        )
    print(f"{'='*len(header)}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents on StockPortfolio-v0")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--n-assets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None, help="Path to trained PPO model")
    args = parser.parse_args()

    env = StockPortfolioEnv(n_assets=args.n_assets, seed=args.seed)
    n = args.n_assets

    agents = [
        (UniformAgent(n), "Equal-Weight"),
        (CashOnlyAgent(n), "Cash-Only"),
        (MomentumAgent(n), "Momentum"),
        (MeanVarianceAgent(n), "Mean-Variance"),
        (RandomAgent(n, seed=args.seed), "Random"),
    ]

    if args.model:
        try:
            from agents.ppo_agent import load_ppo

            ppo = load_ppo(args.model, env=env)
            agents.append((ppo, "PPO (trained)"))
        except Exception as e:
            print(f"Could not load model: {e}")

    all_results = []
    for agent, name in agents:
        print(f"Evaluating {name}...")
        results = evaluate_agent(env, agent, n_episodes=args.episodes, name=name)
        all_results.append(results)

    # Sort by mean return
    all_results.sort(key=lambda x: x["mean_return"], reverse=True)
    print_results(all_results)


if __name__ == "__main__":
    main()
