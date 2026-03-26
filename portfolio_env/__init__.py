from gymnasium.envs.registration import register

register(
    id="StockPortfolio-v0",
    entry_point="portfolio_env.env:StockPortfolioEnv",
)

from portfolio_env.env import StockPortfolioEnv

__all__ = ["StockPortfolioEnv"]
