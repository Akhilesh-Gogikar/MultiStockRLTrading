# MultiStockRLTrading

Trading multiple stocks using a custom Gym environment and custom neural network policies with Stable-Baselines3.

This repository is now prepared for **intelligent multi-asset alpha research**, including a modular training entrypoint, optional LLM analyst blending, and a new **cross-attention policy** that can reason across assets instead of treating each series independently.

## What is new

### Cross-attention policy for multi-asset alpha
The project now includes `CrossAttentionActorCriticPolicy`, which is designed for the exact workflow you described:
- encode each asset's rolling feature window,
- apply **temporal self-attention** within each asset,
- apply **cross-asset attention** across the asset universe,
- compress the shared representation into PPO actor/critic latents.

This makes the codebase a better starting point for:
- relative-value and rotation strategies,
- sector/index lead-lag modeling,
- market-regime aware allocation,
- alpha generation from inter-asset relationships.

### Research-friendly training CLI
`train.py` is now a configurable script instead of a single hard-coded experiment. You can choose:
- policy architecture,
- total timesteps,
- window size,
- train/eval split,
- model name,
- data directory,
- whether to skip inference.

### Robustness improvements
- Graph/capsule policy code no longer hardcodes `cuda:0` for graph edges.
- Legacy GAT attention now respects the configured time window instead of assuming 12 steps.
- Training saves artifacts into predictable directories (`saved_models/`, `artifacts/`).

## Included policy architectures

The training entrypoint supports the following PPO policies:

1. `custom`
   - the original dense policy head.
2. `gat`
   - the graph/capsule-based architecture.
3. `cross_attention`
   - the recommended starting point for intelligent multi-asset alpha modeling.

## Installation

### Conda environment
Create the environment:

```bash
conda env create -f environment.yml
conda activate multi_stock_rl_trading
```

Capsule layer library is still installed separately:

```bash
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Running experiments

### Recommended: cross-attention policy

```bash
python train.py --policy cross_attention --timesteps 10000 --model-name cross_attention_alpha
```

### Original dense baseline

```bash
python train.py --policy custom --timesteps 10000 --model-name dense_baseline
```

### Graph/capsule variant

```bash
python train.py --policy gat --timesteps 10000 --model-name gat_alpha
```

### Useful flags

```bash
python train.py \
  --policy cross_attention \
  --timesteps 25000 \
  --window-size 24 \
  --train-split 1500 \
  --batch-size 256 \
  --model-name xattn_v1
```

## Data pipeline

The repo expects per-asset files under `history_data/` with at least:
- `datetime`
- `timestamp`
- `name`
- `token`
- OHLCV columns (`open`, `high`, `low`, `close`, `volume`)

`train.py` automatically:
- computes technical indicators,
- merges optional fundamentals,
- aligns all assets into a common panel,
- builds a `MultiStockTradingEnv` observation tensor with shape:

```text
(num_assets, window_size, num_features)
```

That observation layout is what enables the new cross-attention policy to model inter-asset structure directly.

## Optional: LLM technical analyst integration

You can blend model actions with an API-driven technical analyst using OpenAI-compatible chat endpoints.

Set environment variables before running inference/training:

```bash
export LLM_ANALYST_ENABLED=1
export LLM_ANALYST_ENDPOINT="https://api.openai.com/v1/chat/completions"
export LLM_ANALYST_API_KEY="<your_key>"
export LLM_ANALYST_MODEL="gpt-4o-mini"
export LLM_ANALYST_BLEND_WEIGHT=0.25
```

- `LLM_ANALYST_BLEND_WEIGHT=0.0` means RL-only.
- `LLM_ANALYST_BLEND_WEIGHT=1.0` means LLM-only signals.

The LLM is asked to return compact JSON like:

```json
{"RELIANCE": 0.4, "TCS": -0.2}
```

where each score must be in `[-1, 1]`.

## Optional: Add Indian fundamental data

You can enrich the RL state with fundamental factors (P/E, EPS, balance-sheet ratios, etc.) from another repository.

1. Clone your other dataset repository next to this project (for example: `../indian_stock_market_data`), or set an explicit path:
   ```bash
   export FUNDAMENTAL_DATA_REPO=/absolute/path/to/indian_stock_market_data
   ```
2. Ensure the fundamentals repo contains CSV/Parquet files with:
   - one symbol column (supported names include: `symbol`, `ticker`, `name`, `tradingsymbol`)
   - one date column (supported names include: `date`, `datetime`, `report_date`, `fiscal_date`)
   - one or more numeric fundamental columns
3. Run training normally. Fundamental features are auto-discovered, prefixed with `fundamental_`, and merged into each stock's bars by date.

If no fundamentals repo is found, training falls back to the original technical-only feature set.

## Suggested next steps for alpha research

If you want to continue evolving this into a stronger intelligent trading stack, the next high-value additions would be:

1. **Return targets and auxiliary losses**
   - add next-bar / next-day return forecasting heads alongside PPO.
2. **Masking and universe control**
   - dynamically disable illiquid or unavailable assets.
3. **Transaction-cost realism**
   - slippage, borrow costs, turnover penalties, and position limits.
4. **Feature groups**
   - news, fundamentals, macro, sentiment, and market microstructure.
5. **Offline evaluation harness**
   - Sharpe, max drawdown, turnover, hit ratio, exposure, and attribution metrics.

## TensorBoard logs

You can inspect logs with:

```bash
tensorboard --logdir tb_logs
```

## Background articles

This work is part of a series of articles written on Medium on Applied RL:

1. Customized Deep Reinforcement Learning for Algorithmic Trading: https://medium.com/@akhileshgogikar/custom-deep-rl-for-algo-trading-106b1a2daa16
2. Custom RL for Algo Trading — Data Preprocessing: https://medium.com/@akhileshgogikar/applied-rl-data-preprocessing-for-algo-trading-4478251b9676
3. Custom Gym environment for multi-stock RL based Algo trading: https://medium.com/@akhileshgogikar/custom-gym-environment-for-multi-stock-algo-trading-113b07dd445d
4. Customization of RL policies using StableBaselines3: https://medium.com/@akhileshgogikar/applied-rl-customizing-neural-networks-for-rl-policies-a5a9e2cf763e
5. Advanced deep learning customization of neural networks for RL based Algo trading: https://medium.com/@akhileshgogikar/applied-reinforcement-learning-3e73ca771bac
