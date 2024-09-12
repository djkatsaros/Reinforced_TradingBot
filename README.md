# Reinforced_TradingBot

Workspace for building a trading bot based on reinforced "Q" learning and DNN for the Q function



## Repo structrure

There are several scripts. 

finance_env contains the main finance 'gym' class. It provides the environment within which the bot can make trading decisions. It includes methods to retrieve and prepare data, print the environment state, reset the environment, take a step in the environment, as well as some auxiliary methods to keep dimensions consistent.

tradingbot is the 

There are 3 backtesting classes, one form the base (backtestingbase) and the other two inheriting from the first. 

## References

I learned this more or less exclusively from Dr. Yves Hilpisch's excellent book _Artificial Intelligence in Finance, a Python-Based Guide_, published in 2021 by O'Reilly. My workflow was to try and write the code in a mix of from scratch and from copying the code in the book in a fashion that facilitated understanding. The initial commits were **very** similar to what is in the book, owing how often I had to reference the book's code. There have been some minor tweaks as I have continued to experiment, but much of the original credit is owed to Dr. Hilpisch. This repo is mainly an effort to learn some intuition for algorithmic trading, and the ways machine learning can create profit in this space under normal circumstances. 
