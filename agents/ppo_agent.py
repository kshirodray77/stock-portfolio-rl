"""
PPO-based RL agent using Stable-Baselines3.

Provides a thin wrapper for training and evaluation that handles
the Dict observation space via a feature extractor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def _check_sb3():
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required. Install with: "
            "pip install stable-baselines3[extra]"
        )


# ------------------------------------------------------------------
# Custom Feature Extractor for Dict obs
# ------------------------------------------------------------------

if SB3_AVAILABLE:

    class PortfolioExtractor(BaseFeaturesExtractor):
        """Flattens the Dict observation into a single feature vector."""

        def __init__(self, observation_space, features_dim: int = 128):
            super().__init__(observation_space, features_dim)

            prices_shape = observation_space["prices"].shape
            returns_shape = observation_space["returns"].shape
            weights_dim = observation_space["weights"].shape[0]
            portfolio_dim = observation_space["portfolio"].shape[0]

            # CNN for price/return series
            n_channels = prices_shape[1] + returns_shape[1]  # concat assets
            seq_len = prices_shape[0]

            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            conv_out = 64

            self.fc = nn.Sequential(
                nn.Linear(conv_out + weights_dim + portfolio_dim, features_dim),
                nn.ReLU(),
            )

        def forward(self, obs: dict) -> torch.Tensor:
            prices = obs["prices"]  # (B, window, n_assets)
            returns = obs["returns"]
            weights = obs["weights"]
            portfolio = obs["portfolio"]

            # Normalise prices per-batch
            prices = prices / (prices[:, -1:, :] + 1e-8)

            # (B, window, 2*n_assets) → (B, 2*n_assets, window)
            x = torch.cat([prices, returns], dim=-1).permute(0, 2, 1)
            conv_out = self.conv(x).squeeze(-1)  # (B, 64)

            combined = torch.cat([conv_out, weights, portfolio], dim=-1)
            return self.fc(combined)


# ------------------------------------------------------------------
# Training API
# ------------------------------------------------------------------


def train_ppo(
    env,
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    verbose: int = 1,
    save_path: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> "PPO":
    """Train a PPO agent on the portfolio environment."""
    _check_sb3()

    policy_kwargs = dict(
        features_extractor_class=PortfolioExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model


def load_ppo(path: str, env=None) -> "PPO":
    """Load a pre-trained PPO model."""
    _check_sb3()
    return PPO.load(path, env=env)
