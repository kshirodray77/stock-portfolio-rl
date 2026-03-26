"""
Tests for the StockPortfolioEnv environment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from portfolio_env import StockPortfolioEnv
from portfolio_env.market import MarketSimulator
from agents.baselines import UniformAgent, CashOnlyAgent, MomentumAgent, RandomAgent


class TestMarketSimulator:
    def test_gbm_shape(self):
        market = MarketSimulator(n_assets=3, total_steps=100, seed=0)
        assert market._prices.shape == (100, 3)

    def test_prices_positive(self):
        market = MarketSimulator(n_assets=5, total_steps=200, seed=42)
        assert np.all(market._prices > 0)

    def test_window(self):
        market = MarketSimulator(n_assets=3, total_steps=50, seed=0)
        for _ in range(10):
            market.step()
        prices, returns = market.get_window(5)
        assert prices.shape == (5, 3)
        assert returns.shape == (5, 3)

    def test_reset_reproducibility(self):
        m1 = MarketSimulator(n_assets=3, total_steps=50, seed=99)
        m2 = MarketSimulator(n_assets=3, total_steps=50, seed=99)
        np.testing.assert_array_almost_equal(m1._prices, m2._prices)


class TestStockPortfolioEnv:
    def test_reset(self):
        env = StockPortfolioEnv(n_assets=3, seed=0)
        obs, info = env.reset()
        assert obs["prices"].shape == (20, 3)
        assert obs["returns"].shape == (20, 3)
        assert obs["weights"].shape == (4,)
        assert obs["portfolio"].shape == (3,)
        assert np.isclose(obs["weights"][-1], 1.0)  # 100% cash

    def test_step_shape(self):
        env = StockPortfolioEnv(n_assets=3, seed=0)
        obs, _ = env.reset()
        action = np.array([0.3, 0.3, 0.3, 0.1], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert obs["prices"].shape == (20, 3)
        assert isinstance(reward, float)
        assert "portfolio_value" in info

    def test_full_episode(self):
        env = StockPortfolioEnv(n_assets=2, max_steps=50, seed=0)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.array([0.4, 0.4, 0.2], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps == 50

    def test_cash_only(self):
        env = StockPortfolioEnv(n_assets=3, max_steps=10, seed=0)
        obs, _ = env.reset()
        for _ in range(10):
            action = np.zeros(4, dtype=np.float32)
            action[-1] = 1.0  # 100% cash
            obs, reward, term, trunc, info = env.step(action)

        # Cash-only should have zero transaction costs
        assert np.isclose(info["portfolio_value"], env.initial_cash, atol=1.0)

    def test_weights_sum_to_one(self):
        env = StockPortfolioEnv(n_assets=3, seed=0)
        obs, _ = env.reset()
        for _ in range(20):
            action = np.random.rand(4).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            assert np.isclose(obs["weights"].sum(), 1.0, atol=0.01)

    def test_reward_modes(self):
        for mode in ["log_return", "sharpe", "sortino"]:
            env = StockPortfolioEnv(n_assets=2, max_steps=30, reward_mode=mode, seed=0)
            obs, _ = env.reset()
            action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            _, reward, _, _, _ = env.step(action)
            assert isinstance(reward, float)
            assert np.isfinite(reward)

    def test_render_ansi(self):
        env = StockPortfolioEnv(n_assets=2, render_mode="ansi", seed=0)
        obs, _ = env.reset()
        env.step(np.array([0.5, 0.3, 0.2]))
        output = env.render()
        assert isinstance(output, str)
        assert "Step" in output


class TestBaselines:
    def test_uniform_agent(self):
        agent = UniformAgent(3)
        obs = {"returns": np.zeros((20, 3))}
        action = agent.act(obs)
        assert action.shape == (4,)
        assert np.isclose(action.sum(), 1.0)

    def test_cash_agent(self):
        agent = CashOnlyAgent(3)
        action = agent.act({})
        assert action[-1] == 1.0
        assert action[:-1].sum() == 0.0

    def test_momentum_agent(self):
        agent = MomentumAgent(3)
        obs = {"returns": np.random.randn(20, 3).astype(np.float32)}
        action = agent.act(obs)
        assert np.isclose(action.sum(), 1.0, atol=0.01)

    def test_random_agent(self):
        agent = RandomAgent(3, seed=0)
        action = agent.act({})
        assert action.shape == (4,)
        assert np.isclose(action.sum(), 1.0, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
