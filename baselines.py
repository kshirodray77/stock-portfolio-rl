"""
Baseline agents for benchmarking against trained RL policies.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class UniformAgent:
    """Equal-weight allocation across all assets (no cash)."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def act(self, obs: dict) -> np.ndarray:
        w = np.ones(self.n_assets + 1, dtype=np.float32)
        w[-1] = 0.0  # no cash
        return w / w.sum()


class CashOnlyAgent:
    """100% cash — the most conservative baseline."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def act(self, obs: dict) -> np.ndarray:
        w = np.zeros(self.n_assets + 1, dtype=np.float32)
        w[-1] = 1.0
        return w


class MomentumAgent:
    """Allocate more to assets with positive recent momentum."""

    def __init__(self, n_assets: int, lookback: int = 5):
        self.n_assets = n_assets
        self.lookback = lookback

    def act(self, obs: dict) -> np.ndarray:
        returns = obs["returns"]  # (window, n_assets)
        recent = returns[-self.lookback :]
        momentum = recent.mean(axis=0)

        # Softmax-style allocation
        pos = np.maximum(momentum, 0)
        total = pos.sum()
        w = np.zeros(self.n_assets + 1, dtype=np.float32)
        if total > 1e-8:
            w[: self.n_assets] = pos / total * 0.9
            w[-1] = 0.1
        else:
            w[-1] = 1.0  # no positive momentum → hold cash
        return w


class MeanVarianceAgent:
    """Simple mean-variance optimisation from rolling returns window."""

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

    def act(self, obs: dict) -> np.ndarray:
        returns = obs["returns"]  # (window, n_assets)
        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False)

        # Regularise
        cov += np.eye(self.n_assets) * 1e-6

        try:
            inv_cov = np.linalg.inv(cov)
            raw_w = inv_cov @ mu / self.risk_aversion
        except np.linalg.LinAlgError:
            raw_w = np.ones(self.n_assets) / self.n_assets

        # Clip negative weights, add cash
        raw_w = np.maximum(raw_w, 0)
        total = raw_w.sum()
        w = np.zeros(self.n_assets + 1, dtype=np.float32)
        if total > 1e-8:
            w[: self.n_assets] = raw_w / total * 0.95
            w[-1] = 0.05
        else:
            w[-1] = 1.0
        return w


class RandomAgent:
    """Random Dirichlet-sampled allocation."""

    def __init__(self, n_assets: int, seed: Optional[int] = None):
        self.n_assets = n_assets
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        w = self._rng.dirichlet(np.ones(self.n_assets + 1)).astype(np.float32)
        return w
