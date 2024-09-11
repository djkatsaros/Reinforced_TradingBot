#
#
# Trading Bot - Financial Q-learning Agent
#
# @ Dr. Yves J. Hippisch
#
#
import os
import random
import numpy as np
from pylab import plt, mpl
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

os.environ['PYTHONHASHSEED'] = '0'  
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

def set_seeds(seed=100):
    ''' Sets seeds for all random number generators'''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
  
class TradingBot:
    def __init__(self, hidden_units, learning_rate, dropout_rate, learn_env,
                 valid_env=None, val=True, dropout=False):
        self.learn_env = learn_env # call two instances of the 'gym'/envmt. Can have different characteristics for valid vs. learning
        self.valid_env = valid_env
        self.val = val # whether to validate or not essentially 
        self.epsilon = 1.0 # Percentage of the time spent in exploration vs exploitation. Begin in pure exploration generating data to learn from
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99 # decay rate of epsilon. Usually very slow. 
        self.learning_rate = learning_rate # for the classifier.
        self.gamma = 0.5 # for the Q- reward equatoin 
        self.batch_size = 128
        self.max_treward = 0
        self.averages = list()  
        self.trewards = list()
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.memory = deque(maxlen=2000) # collection of previous actions to learn from
        self.model = self._build_model(hidden_units,
                             learning_rate, dropout_rate, dropout)

    def _build_model(self, hu, lr, dr, dropout):
        ''' Method to create the DNN model.
        lr = learning rate
        dr = dropout rate (only needed if dropout=True)
        dropout: boolean dictating whether use dropout layers or not.
        '''
        model = Sequential()
        model.add(Dense(hu, input_shape=(
            self.learn_env.lags, self.learn_env.n_features),
            activation='relu'))
        if dropout:
            model.add(Dropout(dr, seed=100))
        model.add(Dense(hu, activation='relu'))
        if dropout:
            model.add(Dropout(dr, seed=100))
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop(learning_rate=lr)
        )
        return model

    def act(self, state):
        ''' Either 
            exploration [generate a random uninformed action. Happens more often the larger epsilon is. Essentially gives
                        new data from which to learn]
            or 
            exploitation [ exploit the previous exploration  to update classifier]
        '''
        if random.random() <= self.epsilon:
            # explore
            return self.learn_env.action_space.sample()
        # exploit: Use current model. 
        action = self.model.predict(state)[0, 0]
        return np.argmax(action)

    def replay(self):
        ''' Method to retrain the DNN model based on
            batches of memorized experiences.
          Uses th Q-learning algorithm, which takes into 
          account the delayed rewards from previous
          actions.
          Everytime we replay the memory the rate of exploration decays.
        '''
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(
                    self.model.predict(next_state)[0, 0]) # executes the Q-learning function.
            target = self.model.predict(state)
            target[0, 0, action] = reward
            # fit model to explored states (from memorry)
            self.model.fit(state, target, epochs=1,
                           verbose=False)
        # epsilon decays everytime we replay.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        ''' Trains the DNN QL agent.
        calls:
          reset, to get a fresh state
          step, to take a step in the environment based on the action taken
          validate: to validate performance of the DNN
          replay: when memory has grown long enough (bigger than batch size), updates/retrains
          the DNN classifier using the accumulated actions (explorations).
        '''
        for e in range(1, episodes + 1):
            state = self.learn_env.reset()
            state = np.reshape(state, [1, self.learn_env.lags,
                                       self.learn_env.n_features])
            for _ in range(10000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action, patience =15)
                next_state = np.reshape(next_state,
                                        [1, self.learn_env.lags,
                                         self.learn_env.n_features])
                self.memory.append([state, action, reward,
                                    next_state, done])
                state = next_state
                if done:
                    treward = _ + 1
                    self.trewards.append(treward)
                    avg = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance
                    self.averages.append(avg)
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)
                    templ = 'episode: {:2d}/{} | treward: {:4d} | '
                    templ += 'perf: {:5.3f} | av: {:5.1f} | max: {:4d}'
                    print(templ.format(e, episodes, treward, perf,
                                       avg, self.max_treward), end='\r')
                    break
            if self.val:
                self.validate(e, episodes)
            if len(self.memory) > self.batch_size:
                self.replay()
        print()

    def validate(self, e, episodes):
        ''' Method to validate the performance of the
            DNN QL agent.
            Is called when self.val = True
        '''
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags,
                                   self.valid_env.n_features])
        for _ in range(10000):
            action = np.argmax(self.model.predict(state)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action, patience =15)
            state = np.reshape(next_state, [1, self.valid_env.lags,
                                            self.valid_env.n_features])
            if done:
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
                if e % int(episodes / 6) == 0:
                    templ = 71 * '='
                    templ += '\nepisode: {:2d}/{} | VALIDATION | '
                    templ += 'treward: {:4d} | perf: {:5.3f} | eps: {:.2f}\n'
                    templ += 71 * '='
                    print(templ.format(e, episodes, treward,
                                       perf, self.epsilon))
                break

    ####################################################
    #             Plotting methods
    #             With polynom. regressions (deg=3)
    ####################################################

    def plot_treward(agent):
        ''' Plot the total reward
            per training eposiode and regression
            (deg=3)
        '''
        plt.figure(figsize=(10, 6))
        x = range(1, len(agent.averages) + 1)
        y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
        plt.plot(x, agent.averages, label='moving average')
        plt.plot(x, y, 'r--', label='regression')
        plt.xlabel('episodes')
        plt.ylabel('total reward')
        plt.legend()


    def plot_performance(agent):
        ''' Plot the financial gross
            performance per training episode
            and regression
            (deg=3)
        '''
        plt.figure(figsize=(10, 6))
        x = range(1, len(agent.performances) + 1)
        y = np.polyval(np.polyfit(x, agent.performances, deg=3), x)
        plt.plot(x, agent.performances[:], label='training')
        plt.plot(x, y, 'r--', label='regression (train)')
        if agent.val:
            y_ = np.polyval(np.polyfit(x, agent.vperformances, deg=3), x)
            plt.plot(x, agent.vperformances[:], label='validation')
            plt.plot(x, y_, 'r-.', label='regression (valid)')
        plt.xlabel('episodes')
        plt.ylabel('gross performance')
        plt.legend()
