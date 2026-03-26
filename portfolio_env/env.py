"""
StockPortfolioEnv — A Gymnasium-compatible RL environment for portfolio rebalancing.

The agent manages a portfolio of N assets + cash, choosing target allocation
weights at each time step. Realistic transaction costs, slippage, and
risk-adjusted reward shaping are included.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional

from portfolio_env.market import MarketSimulator


class StockPortfolioEnv(gym.Env):
    """
    Observation space
    -----------------
    A dict with:
        prices      : (window, n_assets)  — rolling price window
        returns     : (window, n_assets)  — rolling log-return window
        weights     : (n_assets + 1,)     — current portfolio weights (assets + cash)
        portfolio   : (3,)                — [portfolio_value, cash_balance, step/max_steps]

    Action space
    ------------
    Box(0, 1, shape=(n_assets + 1,))  — target allocation weights (auto-normalised)

    Reward
    ------
    Configurable via `reward_mode`:
        "log_return"   — log(new_value / old_value)
        "sharpe"       — rolling Sharpe ratio of log returns
        "sortino"      — rolling Sortino ratio of log returns
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        market: Optional[MarketSimulator] = None,
        n_assets: int = 5,
        window: int = 20,
        max_steps: int = 252,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        reward_mode: str = "log_return",
        sharpe_window: int = 20,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.window = window
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.reward_mode = reward_mode
        self.sharpe_window = sharpe_window
        self.render_mode = render_mode

        # Market data source
        self.market = market or MarketSimulator(
            n_assets=n_assets, total_steps=max_steps + window + 1, seed=seed
        )

        # Spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_assets + 1,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    low=0, high=np.inf, shape=(window, n_assets), dtype=np.float32
                ),
                "returns": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(window, n_assets), dtype=np.float32
                ),
                "weights": spaces.Box(
                    low=0, high=1, shape=(n_assets + 1,), dtype=np.float32
                ),
                "portfolio": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )

        # Internal state
        self._step = 0
        self._weights = np.zeros(n_assets + 1, dtype=np.float32)
        self._portfolio_value = initial_cash
        self._cash = initial_cash
        self._holdings = np.zeros(n_assets, dtype=np.float64)
        self._return_history: list[float] = []
        self._value_history: list[float] = []

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.market.reset(seed=seed)

        self._step = 0
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0  # start 100 % cash
        self._portfolio_value = self.initial_cash
        self._cash = self.initial_cash
        self._holdings = np.zeros(self.n_assets, dtype=np.float64)
        self._return_history = []
        self._value_history = [self.initial_cash]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # 1. Normalise target weights
        target_weights = self._normalise(action)

        # 2. Advance market → new prices
        old_prices = self.market.current_prices()
        self.market.step()
        new_prices = self.market.current_prices()

        # 3. Update holdings value with new prices
        old_value = self._portfolio_value
        self._portfolio_value = self._cash + np.sum(self._holdings * new_prices)

        # 4. Rebalance to target weights
        target_values = target_weights[: self.n_assets] * self._portfolio_value
        target_cash = target_weights[-1] * self._portfolio_value
        target_holdings = target_values / new_prices

        trade_values = np.abs(target_holdings - self._holdings) * new_prices
        total_trade = np.sum(trade_values)
        cost = total_trade * (self.transaction_cost_pct + self.slippage_pct)

        self._holdings = target_holdings
        self._cash = target_cash - cost
        self._portfolio_value = self._cash + np.sum(self._holdings * new_prices)

        # 5. Compute reward
        log_ret = np.log(self._portfolio_value / max(old_value, 1e-8))
        self._return_history.append(log_ret)
        self._value_history.append(self._portfolio_value)
        reward = self._compute_reward(log_ret)

        # 6. Update weights
        if self._portfolio_value > 0:
            self._weights[: self.n_assets] = (
                (self._holdings * new_prices) / self._portfolio_value
            ).astype(np.float32)
            self._weights[-1] = np.float32(self._cash / self._portfolio_value)
        else:
            self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)

        # 7. Check termination
        self._step += 1
        terminated = self._portfolio_value <= 0
        truncated = self._step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["transaction_cost"] = cost
        info["total_trade_value"] = total_trade
        info["log_return"] = log_ret

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self, action: np.ndarray) -> np.ndarray:
        w = np.clip(action, 0, None).astype(np.float64)
        s = w.sum()
        if s < 1e-8:
            w = np.zeros_like(w)
            w[-1] = 1.0  # default to cash
        else:
            w /= s
        return w

    def _get_obs(self) -> dict:
        prices, returns = self.market.get_window(self.window)
        return {
            "prices": prices.astype(np.float32),
            "returns": returns.astype(np.float32),
            "weights": self._weights.copy(),
            "portfolio": np.array(
                [
                    self._portfolio_value,
                    self._cash,
                    self._step / max(self.max_steps, 1),
                ],
                dtype=np.float32,
            ),
        }

    def _get_info(self) -> dict:
        total_return = self._portfolio_value / self.initial_cash - 1
        return {
            "step": self._step,
            "portfolio_value": self._portfolio_value,
            "total_return": total_return,
            "weights": self._weights.copy(),
        }

    def _compute_reward(self, log_ret: float) -> float:
        if self.reward_mode == "log_return":
            return log_ret
        elif self.reward_mode == "sharpe":
            return self._rolling_sharpe()
        elif self.reward_mode == "sortino":
            return self._rolling_sortino()
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _rolling_sharpe(self) -> float:
        rets = np.array(self._return_history[-self.sharpe_window :])
        if len(rets) < 2:
            return 0.0
        std = rets.std()
        if std < 1e-8:
            return 0.0
        return float(rets.mean() / std)

    def _rolling_sortino(self) -> float:
        rets = np.array(self._return_history[-self.sharpe_window :])
        if len(rets) < 2:
            return 0.0
        downside = rets[rets < 0]
        if len(downside) == 0:
            return float(rets.mean()) * 10  # bonus for no downside
        down_std = downside.std()
        if down_std < 1e-8:
            return 0.0
        return float(rets.mean() / down_std)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())

    def _render_ansi(self) -> str:
        val = self._portfolio_value
        ret = (val / self.initial_cash - 1) * 100
        w_str = " | ".join(
            [f"A{i}:{self._weights[i]:.1%}" for i in range(self.n_assets)]
            + [f"Cash:{self._weights[-1]:.1%}"]
        )
        return (
            f"Step {self._step:>4}/{self.max_steps} | "
            f"Value: ${val:>12,.2f} | Return: {ret:>+7.2f}% | {w_str}"
        )
