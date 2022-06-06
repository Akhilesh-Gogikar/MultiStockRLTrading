from tabnanny import verbose
import talib
import joblib
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import glob
import pandas as pd

from multi_stock_trading_env import MultiStockTradingEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG
from gat_capsule_policy import GATActorCriticPolicy
from custom_rl_policy import CustomActorCriticPolicy
CUDA_LAUNCH_BLOCKING=1 

def add_features(tic_df):

    # Returns in the last t intervals
    for t in range(1, 11):
            tic_df[f'ret{t}min'] = tic_df['close'].div(tic_df['open'].shift(t-1)).sub(1)

    # Simple Moving Average based features

    tic_df['sma'] = talib.SMA(tic_df['close'])

    tic_df['5sma'] = talib.SMA(tic_df['close'], timeperiod=5)

    tic_df['20sma'] = talib.SMA(tic_df['close'], timeperiod=20)

    tic_df['bb_upper'], tic_df['bb_middle'], tic_df['bb_lower'] = talib.BBANDS(tic_df['close'], matype=talib.MA_Type.T3)

    tic_df['bb_sell'] = (tic_df['close'] > tic_df['bb_upper'])*1

    tic_df['bb_buy'] = (tic_df['close'] < tic_df['bb_lower'])*1

    tic_df['bb_squeeze'] = (tic_df['bb_upper'] - tic_df['bb_lower'])/tic_df['bb_middle']

    tic_df['mom'] = talib.MOM(tic_df['close'], timeperiod=10)

    tic_df['adx'] = talib.ADX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=10)

    tic_df['mfi'] = talib.MFI(tic_df['high'], tic_df['low'], tic_df['close'], tic_df['volume'], timeperiod=10)

    tic_df['rsi'] = talib.RSI(tic_df['close'], timeperiod=10)
    

    tic_df['trange'] = talib.TRANGE(tic_df['high'], tic_df['low'], tic_df['close'])


    tic_df['bop'] = talib.BOP(tic_df['open'], tic_df['high'], tic_df['low'], tic_df['close'])

    tic_df['cci'] = talib.CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)

    tic_df['STOCHRSI'] = talib.STOCHRSI(tic_df['close'],timeperiod=14,fastk_period=14,fastd_period=3,fastd_matype=0)[0]

    slowk, slowd = talib.STOCH(tic_df['high'], tic_df['low'], tic_df['close'], fastk_period=14,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)

    macd, macdsignal, macdhist = talib.MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    tic_df['slowk'] = slowk

    tic_df['slowd'] = slowd

    tic_df['macd'] = macd

    tic_df['macdsignal'] = macdsignal

    tic_df['macdhist'] = macdhist

    tic_df['NATR'] = talib.NATR(tic_df['high'].ffill(), tic_df['low'].ffill(), tic_df['close'].ffill())

    tic_df['KAMA'] = talib.KAMA(tic_df['close'], timeperiod=10)

    tic_df['MAMA'], tic_df['FAMA'] = talib.MAMA(tic_df['close'])

    tic_df['MAMA_buy'] = np.where((tic_df['MAMA'] < tic_df['FAMA']), 1, 0)

    tic_df['KAMA_buy'] = np.where((tic_df['close'] < tic_df['KAMA']), 1, 0)

    tic_df['sma_buy'] = np.where((tic_df['close'] < tic_df['5sma']), 1, 0)

    tic_df['maco'] = np.where((tic_df['5sma'] < tic_df['20sma']), 1, 0)

    tic_df['rsi_buy'] = np.where((tic_df['rsi'] < 30), 1, 0)

    tic_df['rsi_sell'] = np.where((tic_df['rsi'] > 70), 1, 0)

    tic_df['macd_buy_sell'] = np.where((tic_df['macd'] < tic_df['macdsignal']), 1, 0)

    return tic_df

directory = 'history_data'

indicators = ['open', 'high', 'low', 'close', 'volume', 'ToD', 'DoW',
       'ret1min', 'ret2min', 'ret3min', 'ret4min', 'ret5min', 'ret6min',
       'ret7min', 'ret8min', 'ret9min', 'ret10min', 'sma', '5sma', '20sma',
       'bb_upper', 'bb_middle', 'bb_lower', 'bb_sell', 'bb_buy', 'bb_squeeze',
       'mom', 'adx', 'mfi', 'rsi', 'trange', 'bop', 'cci', 'STOCHRSI', 'slowk',
       'slowd', 'macd', 'macdsignal', 'macdhist', 'NATR', 'KAMA', 'MAMA',
       'FAMA', 'MAMA_buy', 'KAMA_buy', 'sma_buy', 'maco', 'rsi_buy',
       'rsi_sell', 'macd_buy_sell']

dfs = pd.DataFrame()

num_assets = 0

names = []

data_files = glob.iglob(f'.\{directory}/*')

for filename in data_files:

        df = pd.read_csv(filename)

        df['datetime'] = pd.to_datetime(df['datetime'])

        name = df['name'].iloc[0]

        names.append(name)

        # Adding the Time of Day and Day of Week features

        df['ToD'] = df['datetime'].dt.hour + df['datetime'].dt.minute/60
        df['DoW'] = df['datetime'].dt.weekday/6
        df.sort_values(['timestamp'], inplace=True)


        updated_df = add_features(df)

        updated_df['datetime'] = pd.to_datetime(updated_df['datetime'])
        updated_df = df.set_index(pd.DatetimeIndex(updated_df['datetime']))


        updated_df.drop(['timestamp','name','token'], axis=1, inplace=True)

        updated_df.replace([np.inf, -np.inf], 0, inplace=True)

        dfs = pd.concat([dfs,updated_df], axis=1)

        num_assets += 1

dfs.interpolate(method='pad', limit_direction='forward', inplace=True)

print(dfs.columns)

cols_per_asset = int(len(dfs.columns)/num_assets)

df_list = []
price_df = pd.DataFrame()

for i in range(num_assets):
    df = dfs.iloc[:,i*cols_per_asset:i*cols_per_asset+cols_per_asset]
    #print(df.columns)
    df.drop(['datetime'], axis=1, inplace=True)
    price_df[names[i]] = df['close']
    df_list.append(df)

cols_per_asset -= 1


env = MultiStockTradingEnv(df_list,
        price_df,
        num_stocks=num_assets,
        initial_amount=1000000,
        trade_cost=0,
        num_features=cols_per_asset,
        window_size=12,
        frame_bound = (12,len(price_df)-1500),
        tech_indicator_list=indicators)

prices, features = env.process_data()


model = PPO(CustomActorCriticPolicy, env, verbose=2,tensorboard_log='tb_logs', batch_size=256)
# model = PPO('MlpPolicy', env, verbose=2,tensorboard_log='tb_logs', batch_size=256)

# model = PPO(GATActorCriticPolicy, env, verbose=2,tensorboard_log='tb_logs', batch_size=16)

model.learn(total_timesteps=1000)

plt.figure(figsize=(16, 6))
# env.render_all()
# plt.savefig('rewards.jpg')

name="MultiStockTrader"

model.save("saved_models/"+name)

scalers = env.scalers

del model

del env

env = MultiStockTradingEnv(df_list,
        price_df,
        num_stocks=num_assets,
        initial_amount=1000000,
        trade_cost=0,
        num_features=cols_per_asset,
        window_size=12,
        scalers=scalers,
        frame_bound = (len(price_df)-1500,len(price_df)),
        tech_indicator_list=indicators)

model = PPO.load("saved_models/"+name)
prices, features = env.process_data()
obs = env.reset()
count=0
total_rewards = 0
infer_rewards = []
while True: 
        # obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        count+=1
        obs, rewards, done, info = env.step(action)
        # print(action, rewards)
        total_rewards += rewards

        infer_rewards.append(rewards)
        if done:
                print("info", count,info)
                break

print("Total profit: \n", sum(infer_rewards))

infer_steps = price_df.index[len(price_df)-len(infer_rewards):len(price_df)]#np.array(list(range(len(infer_rewards))))
infer_rewards = np.cumsum(np.array(infer_rewards))
sensex_values = env.representative[-len(infer_steps):]

plt.title(name)
plt.plot(infer_steps, infer_rewards, color="red", label='Profit')
plt.plot(infer_steps, sensex_values, color="blue", label='Index')
plt.legend(loc="upper left")
plt.savefig('Infer_rewards.jpg')



