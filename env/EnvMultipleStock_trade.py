import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from config import config


def _sanitize_array(x):
    arr = np.array(x, dtype=np.float64)
    # Avoid turning inf into huge finite numbers (nan_to_num default behavior).
    # Huge magnitudes can destabilize some policies (PPO/A2C) and yield NaN actions.
    arr[~np.isfinite(arr)] = 0.0
    # Clamp extreme magnitudes as an additional safeguard.
    arr = np.clip(arr, -1e12, 1e12)
    return arr

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = int(getattr(config, 'STOCK_DIM', 30))
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0,turbulence_threshold=140
                 ,initial=True, previous_state=[], model_name='', iteration=''):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        state_dim = 1 + STOCK_DIM * 6
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_dim,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold

        # T+3 settlement and price limits
        self.pending_cash_queue = []
        self.settlement_delay = int(getattr(config, 'SETTLEMENT_DELAY_DAYS', 3))
        self.price_limit_percent = float(getattr(config, 'PRICE_LIMIT_PERCENT', 0.07))

        if not self.initial and self.previous_state:
            if isinstance(self.previous_state, tuple):
                prev_state, prev_queue, prev_day = self.previous_state
                self.state = prev_state
                for settle_day, cash in prev_queue:
                    new_settle_day = settle_day - (prev_day + 1)
                    self.pending_cash_queue.append((new_settle_day, cash))
            else:
                self.state = self.previous_state
        else:
            # initalize state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist() + \
                          self.data.cci.values.tolist() + \
                          self.data.adx.values.tolist()
        self.state = list(_sanitize_array(self.state))
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration

    def _get_pending_cash(self):
        return sum(cash for day, cash in self.pending_cash_queue)


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        price = self.state[index+1]
        if (not np.isfinite(price)) or price <= 0:
            return
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+STOCK_DIM+1] > 0:
                shares_to_sell = min(abs(action), self.state[index+STOCK_DIM+1])
                sell_proceeds = price * shares_to_sell * (1 - TRANSACTION_FEE_PERCENT)
                
                # Put proceeds into the pending cash queue
                self.pending_cash_queue.append((self.day + self.settlement_delay, sell_proceeds))

                self.state[index+STOCK_DIM+1] -= shares_to_sell
                self.cost += price * shares_to_sell * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+STOCK_DIM+1] > 0:
                shares_to_sell = self.state[index+STOCK_DIM+1]
                sell_proceeds = price * shares_to_sell * (1 - TRANSACTION_FEE_PERCENT)
                
                # Put proceeds into the pending cash queue
                self.pending_cash_queue.append((self.day + self.settlement_delay, sell_proceeds))

                self.state[index+STOCK_DIM+1] = 0
                self.cost += price * shares_to_sell * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            price = self.state[index+1]
            if (not np.isfinite(price)) or price <= 0:
                return
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += min(available_amount, action)
            
            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            # if turbulence goes over threshold, just stop buying
            pass
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + self._get_pending_cash() + \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))           

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0] + self._get_pending_cash() + sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0] ))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            daily_std = df_total_value['daily_return'].std()
            if (not np.isfinite(daily_std)) or daily_std == 0:
                sharpe = 0
            else:
                sharpe = (4**0.5)*df_total_value['daily_return'].mean()/daily_std
            print("Sharpe: ",sharpe)
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            # Release settled cash
            settled_cash = 0
            remaining_queue = []
            for day_avail, cash in self.pending_cash_queue:
                if self.day >= day_avail:
                    settled_cash += cash
                else:
                    remaining_queue.append((day_avail, cash))
            self.state[0] += settled_cash
            self.pending_cash_queue = remaining_queue

            actions = np.array(actions).reshape(-1)
            actions = _sanitize_array(actions)
            actions = np.clip(actions, -1, 1)
            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
                
            # Price limits (+/- 7%)
            if self.day > 0:
                prev_prices = self.df.loc[self.day - 1, :].adjcp.values
                curr_prices = self.df.loc[self.day, :].adjcp.values
                for idx in range(STOCK_DIM):
                    price_change = (curr_prices[idx] - prev_prices[idx]) / prev_prices[idx]
                    # Floor limit: block selling if price dropped by >= limit
                    if price_change <= -self.price_limit_percent + 0.001 and actions[idx] < 0:
                        actions[idx] = 0
                    # Ceiling limit: block buying if price rose by >= limit
                    elif price_change >= self.price_limit_percent - 0.001 and actions[idx] > 0:
                        actions[idx] = 0

            begin_total_asset = self.state[0] + self._get_pending_cash() + \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
            #print(self.turbulence)
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0] + self._get_pending_cash() + \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING


        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False 
            #self.iteration=self.iteration
            self.rewards_memory = []
            self.pending_cash_queue = []
            #initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
            self.state = list(_sanitize_array(self.state))
        else:
            if isinstance(self.previous_state, tuple):
                prev_state, prev_queue, prev_day = self.previous_state
            else:
                prev_state = self.previous_state
                prev_queue = []
                prev_day = 0

            self.pending_cash_queue = []
            for settle_day, cash in prev_queue:
                new_settle_day = settle_day - (prev_day + 1)
                self.pending_cash_queue.append((new_settle_day, cash))

            previous_total_asset = prev_state[0] + self._get_pending_cash() + \
            sum(np.array(prev_state[1:(STOCK_DIM+1)])*np.array(prev_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory = [previous_total_asset]
            #self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False 
            #self.iteration=iteration
            self.rewards_memory = []
            #initiate state
            #self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
            #[0]*STOCK_DIM + \

            self.state = [ prev_state[0]] + \
                          self.data.adjcp.values.tolist() + \
                          prev_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]+ \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
            self.state = list(_sanitize_array(self.state))
            
        return self.state
    
    def render(self, mode='human',close=False):
        return (self.state, self.pending_cash_queue, self.day)
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]