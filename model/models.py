# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_GAIL(env_train, model_name, timesteps=1000):
    """GAIL Model"""
    #from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        # For evaluation/trading we want deterministic actions.
        # PPO2/A2C default to deterministic=False, which samples actions and can
        # make results noisy or consistently poor due to extra churn/fees.
        action, _states = model.predict(obs_trade, deterministic=True)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        # Deterministic evaluation for consistent sharpe estimation.
        action, _states = model.predict(test_obs, deterministic=True)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    daily_std = df_total_value['daily_return'].std()
    if (not np.isfinite(daily_std)) or daily_std == 0:
        return 0.0
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / daily_std
    if not np.isfinite(sharpe):
        return 0.0
    return sharpe


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []
    last_state_a2c = []
    last_state_ppo = []
    last_state_ddpg = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    train_start = getattr(config, 'TRAIN_START_DATE', 20090000)
    trade_start = getattr(config, 'TRADE_START_DATE', unique_trade_date[0])

    insample_turbulence = df[(df.datadate < trade_start) & (df.datadate >= train_start)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    # For some datasets, the first ~252 days of turbulence are 0 by construction.
    # Including these zeros can make the threshold too low and block trading.
    # We slice off the first 252 days (warmup period) to prevent this bias.
    insample_turb_values = insample_turbulence.turbulence.values
    if len(insample_turb_values) > 252:
        insample_turb_values = insample_turb_values[252:]
    insample_turb_values = insample_turb_values[np.isfinite(insample_turb_values)]
    positive_vals = insample_turb_values[insample_turb_values > 0]
    if len(positive_vals) > 0:
        insample_turb_values = positive_vals
    # Match the original approach (0.90) but exclude warmup zeros to avoid thresholds that are too low.
    insample_turbulence_threshold = np.quantile(insample_turb_values, 0.90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=train_start, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation data (we will create one env per model to save outputs separately)
        validation = data_split(
            df,
            start=unique_trade_date[i - rebalance_window - validation_window],
            end=unique_trade_date[i - rebalance_window],
        )
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", train_start, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        env_val_a2c = DummyVecEnv([lambda: StockEnvValidation(
            validation,
            turbulence_threshold=turbulence_threshold,
            iteration=f"{i}_A2C",
        )])
        obs_val_a2c = env_val_a2c.reset()
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val_a2c, test_obs=obs_val_a2c)
        sharpe_a2c = get_validation_sharpe(f"{i}_A2C")
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        env_val_ppo = DummyVecEnv([lambda: StockEnvValidation(
            validation,
            turbulence_threshold=turbulence_threshold,
            iteration=f"{i}_PPO",
        )])
        obs_val_ppo = env_val_ppo.reset()
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val_ppo, test_obs=obs_val_ppo)
        sharpe_ppo = get_validation_sharpe(f"{i}_PPO")
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10000)
        #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        env_val_ddpg = DummyVecEnv([lambda: StockEnvValidation(
            validation,
            turbulence_threshold=turbulence_threshold,
            iteration=f"{i}_DDPG",
        )])
        obs_val_ddpg = env_val_ddpg.reset()
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val_ddpg, test_obs=obs_val_ddpg)
        sharpe_ddpg = get_validation_sharpe(f"{i}_DDPG")

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        # Save individual model trading results for analysis
        last_state_a2c = DRL_prediction(
            df=df,
            model=model_a2c,
            name="A2C",
            last_state=last_state_a2c,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )
        last_state_ppo = DRL_prediction(
            df=df,
            model=model_ppo,
            name="PPO",
            last_state=last_state_ppo,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )
        last_state_ddpg = DRL_prediction(
            df=df,
            model=model_ddpg,
            name="DDPG",
            last_state=last_state_ddpg,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )

        # Ensemble trading (selected best model)
        last_state_ensemble = DRL_prediction(
            df=df,
            model=model_ensemble,
            name="ensemble",
            last_state=last_state_ensemble,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
