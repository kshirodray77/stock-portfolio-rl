"""
MarketSimulator — generates synthetic multi-asset price series.

Supports two modes:
    "gbm"       — correlated Geometric Brownian Motion (default)
    "historical" — replay from a pandas DataFrame of prices

The simulator pre-generates an entire episode on reset() so that
price look-ups are O(1).
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import warnings


class MarketSimulator:
    """Synthetic or historical market price generator."""

    def __init__(
        self,
        n_assets: int = 5,
        total_steps: int = 300,
        mode: str = "gbm",
        # GBM parameters
        annual_mu_range: tuple[float, float] = (0.02, 0.15),
        annual_sigma_range: tuple[float, float] = (0.10, 0.40),
        correlation_strength: float = 0.3,
        # Historical replay
        price_df=None,  # pandas DataFrame (columns = assets)
        seed: Optional[int] = None,
    ):
        self.n_assets = n_assets
        self.total_steps = total_steps
        self.mode = mode
        self.annual_mu_range = annual_mu_range
        self.annual_sigma_range = annual_sigma_range
        self.correlation_strength = correlation_strength
        self.price_df = price_df

        self._rng = np.random.default_rng(seed)
        self._prices: np.ndarray = np.empty(0)  # (total_steps, n_assets)
        self._log_returns: np.ndarray = np.empty(0)
        self._cursor = 0

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cursor = 0

        if self.mode == "gbm":
            self._prices = self._generate_gbm()
        elif self.mode == "historical":
            self._prices = self._load_historical()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Pre-compute log returns
        self._log_returns = np.diff(np.log(self._prices), axis=0)  # (T-1, n)

    def step(self) -> None:
        self._cursor += 1
        if self._cursor >= len(self._prices):
            warnings.warn("MarketSimulator stepped past available data.")
            self._cursor = len(self._prices) - 1

    def current_prices(self) -> np.ndarray:
        return self._prices[self._cursor].copy()

    def get_window(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (prices, log_returns) of shape (window, n_assets)."""
        end = self._cursor + 1
        start = max(0, end - window)

        prices = self._prices[start:end]
        if len(prices) < window:
            pad = np.tile(prices[0], (window - len(prices), 1))
            prices = np.vstack([pad, prices])

        ret_end = self._cursor
        ret_start = max(0, ret_end - window)
        returns = self._log_returns[ret_start:ret_end]
        if len(returns) < window:
            pad = np.zeros((window - len(returns), self.n_assets))
            returns = np.vstack([pad, returns])

        return prices, returns

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def _generate_gbm(self) -> np.ndarray:
        n = self.n_assets
        T = self.total_steps
        dt = 1 / 252  # daily

        mu = self._rng.uniform(*self.annual_mu_range, size=n)
        sigma = self._rng.uniform(*self.annual_sigma_range, size=n)
        initial = self._rng.uniform(20, 200, size=n)

        # Build correlation matrix
        corr = np.full((n, n), self.correlation_strength)
        np.fill_diagonal(corr, 1.0)
        D = np.diag(sigma * np.sqrt(dt))
        cov = D @ corr @ D

        L = np.linalg.cholesky(cov)
        Z = self._rng.standard_normal((T - 1, n))
        corr_shocks = Z @ L.T

        drift = (mu - 0.5 * sigma**2) * dt

        log_prices = np.zeros((T, n))
        log_prices[0] = np.log(initial)
        for t in range(1, T):
            log_prices[t] = log_prices[t - 1] + drift + corr_shocks[t - 1]

        # Add occasional regime shifts for realism
        for asset in range(n):
            n_regimes = self._rng.integers(0, 3)
            for _ in range(n_regimes):
                shift_point = self._rng.integers(T // 4, 3 * T // 4)
                shift_magnitude = self._rng.normal(0, 0.02)
                log_prices[shift_point:, asset] += shift_magnitude

        return np.exp(log_prices)

    def _load_historical(self) -> np.ndarray:
        if self.price_df is None:
            raise ValueError("price_df required for 'historical' mode")
        data = self.price_df.values[: self.total_steps].astype(np.float64)
        self.n_assets = data.shape[1]
        return data
