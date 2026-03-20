import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from stable_baselines3 import PPO

from cross_attention_policy import CrossAttentionActorCriticPolicy
from custom_rl_policy import CustomActorCriticPolicy
from fundamental_data import FundamentalDataIntegrator
from gat_capsule_policy import GATActorCriticPolicy
from llm_analyst import LLMAnalystConfig, LLMTechnicalAnalyst, blend_actions
from multi_stock_trading_env import MultiStockTradingEnv


DEFAULT_DATA_DIRECTORY = "history_data"
DEFAULT_WINDOW_SIZE = 12
DEFAULT_TRAINING_SPLIT = 1500
DEFAULT_INITIAL_AMOUNT = 1_000_000
DEFAULT_MODEL_DIR = Path("saved_models")
DEFAULT_FIGURE_DIR = Path("artifacts")


def add_features(tic_df: pd.DataFrame) -> pd.DataFrame:
    enriched = tic_df.copy()

    for t in range(1, 11):
        enriched[f"ret{t}min"] = enriched["close"].div(enriched["open"].shift(t - 1)).sub(1)

    enriched["sma"] = talib.SMA(enriched["close"])
    enriched["5sma"] = talib.SMA(enriched["close"], timeperiod=5)
    enriched["20sma"] = talib.SMA(enriched["close"], timeperiod=20)

    enriched["bb_upper"], enriched["bb_middle"], enriched["bb_lower"] = talib.BBANDS(
        enriched["close"], matype=talib.MA_Type.T3
    )
    enriched["bb_sell"] = (enriched["close"] > enriched["bb_upper"]) * 1
    enriched["bb_buy"] = (enriched["close"] < enriched["bb_lower"]) * 1
    enriched["bb_squeeze"] = (enriched["bb_upper"] - enriched["bb_lower"]) / enriched["bb_middle"]

    enriched["mom"] = talib.MOM(enriched["close"], timeperiod=10)
    enriched["adx"] = talib.ADX(enriched["high"], enriched["low"], enriched["close"], timeperiod=10)
    enriched["mfi"] = talib.MFI(enriched["high"], enriched["low"], enriched["close"], enriched["volume"], timeperiod=10)
    enriched["rsi"] = talib.RSI(enriched["close"], timeperiod=10)
    enriched["trange"] = talib.TRANGE(enriched["high"], enriched["low"], enriched["close"])
    enriched["bop"] = talib.BOP(enriched["open"], enriched["high"], enriched["low"], enriched["close"])
    enriched["cci"] = talib.CCI(enriched["high"], enriched["low"], enriched["close"], timeperiod=14)
    enriched["STOCHRSI"] = talib.STOCHRSI(
        enriched["close"], timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0
    )[0]

    slowk, slowd = talib.STOCH(
        enriched["high"],
        enriched["low"],
        enriched["close"],
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    macd, macdsignal, macdhist = talib.MACD(enriched["close"], fastperiod=12, slowperiod=26, signalperiod=9)

    enriched["slowk"] = slowk
    enriched["slowd"] = slowd
    enriched["macd"] = macd
    enriched["macdsignal"] = macdsignal
    enriched["macdhist"] = macdhist
    enriched["NATR"] = talib.NATR(enriched["high"].ffill(), enriched["low"].ffill(), enriched["close"].ffill())
    enriched["KAMA"] = talib.KAMA(enriched["close"], timeperiod=10)
    enriched["MAMA"], enriched["FAMA"] = talib.MAMA(enriched["close"])

    enriched["MAMA_buy"] = np.where(enriched["MAMA"] < enriched["FAMA"], 1, 0)
    enriched["KAMA_buy"] = np.where(enriched["close"] < enriched["KAMA"], 1, 0)
    enriched["sma_buy"] = np.where(enriched["close"] < enriched["5sma"], 1, 0)
    enriched["maco"] = np.where(enriched["5sma"] < enriched["20sma"], 1, 0)
    enriched["rsi_buy"] = np.where(enriched["rsi"] < 30, 1, 0)
    enriched["rsi_sell"] = np.where(enriched["rsi"] > 70, 1, 0)
    enriched["macd_buy_sell"] = np.where(enriched["macd"] < enriched["macdsignal"], 1, 0)

    return enriched


BASE_INDICATORS = [
    "open", "high", "low", "close", "volume", "ToD", "DoW",
    "ret1min", "ret2min", "ret3min", "ret4min", "ret5min", "ret6min",
    "ret7min", "ret8min", "ret9min", "ret10min", "sma", "5sma", "20sma",
    "bb_upper", "bb_middle", "bb_lower", "bb_sell", "bb_buy", "bb_squeeze",
    "mom", "adx", "mfi", "rsi", "trange", "bop", "cci", "STOCHRSI", "slowk",
    "slowd", "macd", "macdsignal", "macdhist", "NATR", "KAMA", "MAMA",
    "FAMA", "MAMA_buy", "KAMA_buy", "sma_buy", "maco", "rsi_buy",
    "rsi_sell", "macd_buy_sell",
]

POLICY_REGISTRY = {
    "custom": CustomActorCriticPolicy,
    "gat": GATActorCriticPolicy,
    "cross_attention": CrossAttentionActorCriticPolicy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-asset RL trader with optional cross-attention.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--policy", choices=sorted(POLICY_REGISTRY.keys()), default="cross_attention")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--train-split", type=int, default=DEFAULT_TRAINING_SPLIT)
    parser.add_argument("--initial-amount", type=float, default=DEFAULT_INITIAL_AMOUNT)
    parser.add_argument("--trade-cost", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-name", default="MultiStockTrader")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--tensorboard-log", default="tb_logs")
    parser.add_argument("--fundamental-repo", default=os.getenv("FUNDAMENTAL_DATA_REPO"))
    return parser.parse_args()


def load_market_data(
    data_dir: str,
    fundamental_integrator: FundamentalDataIntegrator,
) -> Tuple[List[pd.DataFrame], pd.DataFrame, List[str], List[str]]:
    dfs = pd.DataFrame()
    names: List[str] = []
    num_assets = 0

    data_files = sorted(glob.glob(f"./{data_dir}/*"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir!r}")

    for filename in data_files:
        df = pd.read_csv(filename)
        df["datetime"] = pd.to_datetime(df["datetime"])
        name = df["name"].iloc[0]
        names.append(name)

        df["ToD"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
        df["DoW"] = df["datetime"].dt.weekday / 6
        df.sort_values(["timestamp"], inplace=True)

        updated_df = add_features(df)
        updated_df = fundamental_integrator.merge_with_price_data(updated_df, symbol=name, datetime_col="datetime")
        updated_df["datetime"] = pd.to_datetime(updated_df["datetime"])
        updated_df = updated_df.set_index(pd.DatetimeIndex(updated_df["datetime"]))
        updated_df.drop(["timestamp", "name", "token"], axis=1, inplace=True)
        updated_df.replace([np.inf, -np.inf], 0, inplace=True)

        dfs = pd.concat([dfs, updated_df], axis=1)
        num_assets += 1

    dfs.interpolate(method="pad", limit_direction="forward", inplace=True)
    cols_per_asset = int(len(dfs.columns) / num_assets)

    indicators = BASE_INDICATORS + fundamental_integrator.feature_names
    df_list: List[pd.DataFrame] = []
    price_df = pd.DataFrame()

    for index in range(num_assets):
        asset_df = dfs.iloc[:, index * cols_per_asset : index * cols_per_asset + cols_per_asset].copy()
        asset_df.drop(["datetime"], axis=1, inplace=True)
        price_df[names[index]] = asset_df["close"]
        df_list.append(asset_df)

    return df_list, price_df, names, indicators


def make_env(
    df_list: List[pd.DataFrame],
    price_df: pd.DataFrame,
    indicators: List[str],
    window_size: int,
    frame_bound: Tuple[int, int],
    initial_amount: float,
    trade_cost: float,
    scalers=None,
) -> MultiStockTradingEnv:
    return MultiStockTradingEnv(
        df_list,
        price_df,
        num_stocks=len(df_list),
        initial_amount=initial_amount,
        trade_cost=trade_cost,
        num_features=len(indicators),
        window_size=window_size,
        frame_bound=frame_bound,
        scalers=scalers,
        tech_indicator_list=indicators,
    )


def run_inference(
    env: MultiStockTradingEnv,
    model: PPO,
    price_df: pd.DataFrame,
    names: List[str],
    llm_config: LLMAnalystConfig,
    llm_analyst: LLMTechnicalAnalyst,
) -> Tuple[np.ndarray, np.ndarray]:
    env.process_data()
    obs = env.reset()
    infer_rewards: List[float] = []

    while True:
        action, _states = model.predict(obs)

        if llm_config.enabled:
            current_row_idx = env.frame_bound[0] - env.window_size + env._current_tick
            latest_row = price_df.iloc[min(current_row_idx, len(price_df) - 1)]
            market_snapshot = {symbol: {"close": float(latest_row[symbol])} for symbol in names}
            llm_scores = llm_analyst.get_signal(names, market_snapshot)
            action = blend_actions(action, llm_scores, names, llm_config.blend_weight)

        obs, rewards, done, info = env.step(action)
        infer_rewards.append(rewards)
        if done:
            print("inference_info", info)
            break

    infer_steps = price_df.index[len(price_df) - len(infer_rewards) : len(price_df)]
    cumulative_rewards = np.cumsum(np.array(infer_rewards))
    return np.array(infer_steps), cumulative_rewards


def main() -> None:
    args = parse_args()
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fundamental_integrator = FundamentalDataIntegrator(repo_path=args.fundamental_repo)
    if fundamental_integrator.is_enabled:
        print(
            f"Loaded {len(fundamental_integrator.feature_names)} fundamental features from: "
            f"{fundamental_integrator.repo_path}"
        )
    else:
        print("No fundamental dataset detected. Using technical-only features.")

    llm_config = LLMAnalystConfig.from_env()
    llm_analyst = LLMTechnicalAnalyst(llm_config)

    df_list, price_df, names, indicators = load_market_data(args.data_dir, fundamental_integrator)
    train_end = len(price_df) - args.train_split
    if train_end <= args.window_size:
        raise ValueError(
            f"Training split leaves too little history. Need more than {args.window_size} rows before holdout, got {train_end}."
        )

    env = make_env(
        df_list=df_list,
        price_df=price_df,
        indicators=indicators,
        window_size=args.window_size,
        frame_bound=(args.window_size, train_end),
        initial_amount=args.initial_amount,
        trade_cost=args.trade_cost,
    )
    env.process_data()

    policy_cls = POLICY_REGISTRY[args.policy]
    model = PPO(
        policy_cls,
        env,
        verbose=2,
        tensorboard_log=args.tensorboard_log,
        batch_size=args.batch_size,
    )
    model.learn(total_timesteps=args.timesteps)

    model_path = DEFAULT_MODEL_DIR / args.model_name
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

    if args.skip_inference:
        return

    eval_env = make_env(
        df_list=df_list,
        price_df=price_df,
        indicators=indicators,
        window_size=args.window_size,
        frame_bound=(train_end, len(price_df)),
        initial_amount=args.initial_amount,
        trade_cost=args.trade_cost,
        scalers=env.scalers,
    )
    loaded_model = PPO.load(str(model_path))
    infer_steps, cumulative_rewards = run_inference(eval_env, loaded_model, price_df, names, llm_config, llm_analyst)

    plt.figure(figsize=(16, 6))
    plt.title(args.model_name)
    plt.plot(infer_steps, cumulative_rewards, color="red", label="Profit")
    plt.legend()
    figure_path = DEFAULT_FIGURE_DIR / f"{args.model_name}_inference.png"
    plt.savefig(figure_path)
    print(f"Saved inference plot to {figure_path}")


if __name__ == "__main__":
    main()
