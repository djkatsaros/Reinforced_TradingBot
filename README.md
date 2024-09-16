# Reinforced_TradingBot

Workspace for building a trading bot based on reinforced "Q" learning and DNN for the Q function

## Reinforced Q-learning framework. 

A general setup for reinforcement learning would be an _agent_, a _set of states_, and a _set of actions per state_. The action on a particular state generates a _reward_ for the agent, which tries to maximize the reward. The agent is given functionality to compute future rewards so as to influence it's current decision. This involves the 
_Quality_ function _Q_, from which the algorithm derives it's name. 


## Repo Contents

There are several scripts.  

finance_env contains the main finance 'gym' class. It provides the environment within which the bot can make trading decisions. It includes methods to retrieve and prepare data, print the environment state, reset the environment, take a step in the environment, as well as some auxiliary methods to keep dimensions consistent. There is a set parameter intraday which when set to True means intraday data is pulled by the get_data method. If False, EOD data is returned by the get_data method instead. The step method is important, and does most of the main functions of the environment. It checks whether the agent has taken the optimal action, computes the current return [indexed by bar], and computes/records the return based and right/wrong based rewards. It then pushes the environment forewards and appends the cumulative accuracy and performance stats. Also important is the stopping conditions. The agent stops if we exhaust the dataset or if performance drops too low below  the set min_performance (up to the patience parameter).

tradingbot is the actual Q-learning agent which navigates the environment, making trading decisions as it goes. Includes methods to take an action in the environment, based on the exploration vs. exploitation dynamic. That is, there is a probability (which increases everytime the agent learns from it's actions in the replay method) that 
the agent will bank a memory of how it's action panned out, and a probability it will instead take an action _based_ on those memories (exploitation). This is implemented by first calling the act method to get a ac action for a state (from calling reset). that action is either an essentially random state (explore) or a prediction by the current model (exploit).  In the learn method, where act is called, a step is then taken in the environment based on the returned action, and this is recorded as the next_state. Each call of the step function returns a done value according to the stopping conditions, and if done=True the learn function banks the total reward and some other performance statistics. Eventually, once we have learned alot from the memory banks, the exploration occurs only with probability 'epsilon_min' (=0.1, say).  
there are also replay and validate methods. The replay method serves to help the Q-learning agent (deep dense NN) learn from the banked memory. This is where the Q-learning equation is executed. This serves to update the reward, which then is used to fit the most recent state on the model, updating the model. Vaidate is called id self.val = True. Validate has similar functionality as the learn method, but without the updates to the reward stats.

There are 3 backtesting classes, one form the base (backtestingbase) and the other two inheriting from the first. The backtestingbase script contains most of the functionality. This is primarily the methods to place buy, sell or close out orders (the main actions the agent may take). There are also methods to print the net welath, balances, get date_price etc. backtestingRM is similar, including only the sell/buy order functions (and may be redundant ultimately, oops...). botbacktestingRM calls backtestingRM (and through it backtestingbase), and adds the backtest_strategy method. The main workhorse actions are towards the bottom of the script. This involes getting the relavent state, checking what position we're currently in (long [1], short [-1] or neutral [0]) and proceeding accordingly from that state. Number of trades are updated etc. The self.wait parameter is similar to patience in that it provides some lag time before making trading decisions. 


## References

Cool thinking I've tried to incorporate:
http://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf

I learned this more or less exclusively from Dr. Yves Hilpisch's excellent book _Artificial Intelligence in Finance, a Python-Based Guide_, published in 2021 by O'Reilly. My workflow was to try and write the code in a mix of from scratch and from copying the code in the book in a fashion that facilitated understanding. The initial commits were **very** similar to what is in the book, owing how often I had to reference the book's code. There have been some minor tweaks as I have continued to experiment, but much of the original credit is owed to Dr. Hilpisch. This repo is mainly an effort to learn some intuition for algorithmic trading, and the ways machine learning can create profit in this space under normal circumstances. 

