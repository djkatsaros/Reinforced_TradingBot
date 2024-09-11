#
# Finance environment
#
# @ Dr. Yves J. Hippisch
#
#
import math
import random
import numpy as np
import pandas as pd

class observation_space:
    def __init__(self,n):
        self.shape = (n,)

class action_space:
    def __init__(self,n):
        self.n = n
    
    def sample(self):
        return random.randint(0, self.n - 1)

class Finance:
    intraday = False
    #time_increment = '30min'
    uRl = True 
    if uRl:
        if intraday:
            url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'
        else:
            url = 'http://hilpisch.com/aiif_eikon_eod_data.csv'
    
    def __init__(self, symbol, features, window, lags,
                leverage = 1, min_performance = 0.85, min_accuracy = 0.5,
                start = 0, end=None, mu=None, std=None, time_increment='30min'):
        self.symbol = symbol
        self.features = features # features which define the state
        self.n_features = len(features) # number of features 
        self.window = window
        self.lags = lags # number of lags
        self.leverage = leverage # not 1 if , for example, one is borrowing capital to invest with (higher risk, higher reward).
        self.min_performance = min_performance # minimum (gross) perf required 
        self.min_accuracy = min_accuracy # minimum accuracy
        self.start = start # starting and ending increments in the dataset.
        self.end = end
        self.mu = mu
        self.std = std
        self.observation_space = observation_space(self.lags)
        self.action_space = action_space(2)
        self._get_data(time_increment)
        self._prepare_data()
        
    def _get_data(self, _increment):
        #TODO: allow data from alpha V, or web scraping.
        self.raw = pd.read_csv(self.url, index_col=0, parse_dates=True).dropna()
        
        if self.intraday:
            self.raw = self.raw = self.raw.resample(_increment, label='right').last()
            self.raw = pd.DataFrame(self.raw['Close'])
            self.raw.columns = [self.symbol]
    
    def _prepare_data(self):
        self.data = pd.DataFrame(self.raw[self.symbol])
        self.data = self.data.iloc[self.start:]
        self.data['r'] = np.log(self.data / self.data.shift(1)) # returns
        self.data.dropna(inplace = True)
        self.data['s'] = self.data[self.symbol].rolling(self.window).mean() # SMA
        self.data['m'] = self.data['r'].rolling(self.window).mean() # Momentum
        self.data['v'] = self.data['r'].rolling(self.window).std()  # Variance/volatility
        self.data.dropna(inplace=True) # drop rows with NaN (i.e. first {window} rows before SMA can be computed).
        if self.mu is None:
            # compute mean and variance 
            self.mu = self.data.mean()
            self.std = self.data.std()
        # Perform Gaussian normalization of the data 
        self.data_norm = (self.data - self.mu) / self.std 
        # where returns are positive, set to 1, else set to 0.
        self.data_norm['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data_norm['d'] = self.data_['d'].astype(int)
        if self.end is not None:
            self.data = self.data.iloc[:self.end - self.start]
            self.data_norm = self.data_norm.iloc[:self.end - self.start]
            
    def _get_state(self):
        return self.data_norm[self.features].iloc[self.bar - self.lags:self.bar]
    
    def get_state(self,bar):
        return self.data_norm[self.features].iloc[bar - self.lags:bar]
    
    def seed(self, seed):
      """Method to set random seeds"""
        random.seed(seed)
        np.random.seed(seed)
        
    def reset(self):
      """This method resets the envmt to initial state, setting all the parameters to be baseline."""
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.bar = self.lags
        state = self.data_[self.features].iloc[self.bar -
                                               self.lags:self.bar]
        
        return state.values
    
    def step(self, action, patience):
        """Takes a step in the environment."""
        correct = action == self.data_norm['d'].iloc[self.bar] # check whether agent has made the right action
        ret = self.data['r'].iloc[self.bar] * self.leverage # leveraged return
        reward_1 = 1 if correct else 0
        reward_2 = abs(ret) if correct else -abs(ret) # return based reward
        self.treward += reward_1
        self.bar += 1 # increments the environment forwards
        self.accuracy = self.treward / (self.bar - self.lags) # cumulative accuracy of all trades
        self.performance *= math.exp(reward_2) # gross performance of the taken step
        
        # set stopping conditions
        if self.bar >= len(self.data):
            done = True # end if agent reaches end of dataset. Considered a success.
        elif reward_1 == 1:
            # continues so long as we make a good trade
            done =False
        elif (self.performance < self.min_performance and
             self.bar > self.lags + patience):
            # ends if performance drops below the min performance threshold after a set number of steps (patience)
            done =True 
        else:
            done = False
            
        state = self._get_state()
        info = {}
        return state.values, reward_1 + reward_2 * 5, done, info
      
