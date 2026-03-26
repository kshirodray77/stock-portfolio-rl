# 📈 Stock Portfolio RL Environment

A **Gymnasium-compatible reinforcement learning environment** for training agents to manage and rebalance a multi-asset stock portfolio with realistic transaction costs, slippage, and multiple reward shaping strategies.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Gymnasium](https://img.shields.io/badge/gymnasium-0.29+-green.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)

---

## Overview

The agent manages a portfolio of **N assets + cash**, deciding target allocation weights at each daily time step. The environment models:

- **Correlated asset dynamics** via Geometric Brownian Motion (or historical replay)
- **Transaction costs** — configurable percentage per trade
- **Slippage** — market impact on execution
- **Regime shifts** — occasional structural breaks in price dynamics
- **Three reward modes** — log return, rolling Sharpe ratio, rolling Sortino ratio

## Architecture

```
stock-portfolio-rl/
├── portfolio_env/
│   ├── __init__.py          # Gym registration
│   ├── env.py               # StockPortfolioEnv (core environment)
│   └── market.py            # MarketSimulator (GBM / historical)
├── agents/
│   ├── __init__.py
│   ├── baselines.py         # Equal-weight, Momentum, Mean-Variance, etc.
│   └── ppo_agent.py         # PPO wrapper with custom feature extractor
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Benchmark evaluation
├── tests/
│   └── test_env.py          # Unit tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/stock-portfolio-rl.git
cd stock-portfolio-rl

# Install base (env + baselines only)
pip install -e .

# Install with training support (SB3 + PyTorch)
pip install -e ".[train]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

### 1. Run the environment manually

```python
from portfolio_env import StockPortfolioEnv
import numpy as np

env = StockPortfolioEnv(n_assets=5, reward_mode="sharpe", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(252):  # 1 trading year
    action = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # 5 assets + cash
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

print(f"Final return: {info['total_return']:.2%}")
```

### 2. Benchmark baselines

```bash
python scripts/evaluate.py --episodes 50 --seed 42
```

Output:
```
Agent                Mean %     Std %   Sharpe  MaxDD %   Best %  Worst %
============================================================================
Momentum             +12.34%   8.21%    0.542   14.32%  +28.91%   -3.12%
Mean-Variance         +9.87%   6.54%    0.481   11.20%  +22.45%   -1.98%
Equal-Weight          +7.23%   5.89%    0.395   12.67%  +18.34%   -4.56%
Random                +2.14%  11.32%    0.078   22.45%  +19.87%  -15.23%
Cash-Only             +0.00%   0.00%    0.000    0.00%   +0.00%   +0.00%
```

### 3. Train a PPO agent

```bash
python scripts/train.py --timesteps 500000 --reward sharpe --seed 42
```

### 4. Evaluate trained agent vs baselines

```bash
python scripts/evaluate.py --model checkpoints/ppo_sharpe_500000 --episodes 50
```

## Environment Details

### Observation Space (Dict)

| Key         | Shape                  | Description                             |
|-------------|------------------------|-----------------------------------------|
| `prices`    | `(window, n_assets)`   | Rolling normalised price window         |
| `returns`   | `(window, n_assets)`   | Rolling log-return window               |
| `weights`   | `(n_assets + 1,)`      | Current portfolio weights (incl. cash)  |
| `portfolio` | `(3,)`                 | [value, cash_balance, step_progress]    |

### Action Space

`Box(0, 1, shape=(n_assets + 1,))` — target allocation weights, auto-normalised to sum to 1.

### Reward Modes

| Mode          | Formula                                        | Best for              |
|---------------|------------------------------------------------|-----------------------|
| `log_return`  | `log(V_new / V_old)`                           | Simple, fast training |
| `sharpe`      | Rolling `mean(r) / std(r)` over K steps        | Risk-adjusted returns |
| `sortino`     | Rolling `mean(r) / downside_std(r)` over K steps | Downside protection  |

### Configuration

```python
StockPortfolioEnv(
    n_assets=5,               # Number of tradeable assets
    window=20,                # Observation lookback window
    max_steps=252,            # Episode length (trading days)
    initial_cash=100_000,     # Starting portfolio value
    transaction_cost_pct=0.001,  # 10 bps per trade
    slippage_pct=0.0005,      # 5 bps slippage
    reward_mode="sharpe",     # "log_return" | "sharpe" | "sortino"
)
```

### Using Historical Data

```python
import pandas as pd
from portfolio_env.market import MarketSimulator
from portfolio_env import StockPortfolioEnv

# Load your own price data (columns = assets, rows = daily prices)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

market = MarketSimulator(mode="historical", price_df=prices)
env = StockPortfolioEnv(market=market, n_assets=prices.shape[1])
```

## Baselines

| Agent            | Strategy                                                  |
|------------------|-----------------------------------------------------------|
| `UniformAgent`   | Equal-weight across all assets, no cash                   |
| `CashOnlyAgent`  | 100% cash — conservative benchmark                       |
| `MomentumAgent`  | Allocate to assets with positive recent returns           |
| `MeanVarianceAgent` | Classic Markowitz mean-variance optimisation            |
| `RandomAgent`    | Dirichlet-sampled random weights                          |

## Custom Feature Extractor

The PPO agent uses a custom `PortfolioExtractor` that processes the Dict observation:

1. **1D-CNN** processes the price and return time series → temporal features
2. **Concatenation** with current weights and portfolio state
3. **FC layer** → 128-dim feature vector

This architecture captures both temporal patterns in market data and the agent's current position.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Ideas for Extension

- **Multi-agent** — multiple portfolio managers competing in the same market
- **Options/derivatives** — add options as tradeable instruments
- **Macro features** — include interest rates, VIX, sector indicators as observations
- **Transaction tax** — model stamp duty or Tobin tax
- **Short selling** — allow negative weights with borrowing costs
- **Real-time data** — connect to a live data feed for paper trading

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{stock_portfolio_rl,
  title={Stock Portfolio RL Environment},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/stock-portfolio-rl}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
