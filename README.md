# RL_Trading_with_visual_inputs
Code for Msc. thesis Modelling trading behavior using reinforcement learning with visual market data inputs. Aim of this study is to find out whenever RL agent interprets and reacts to market structures in a similar way as human traders do and whenever additional technical indicators change agents behavior.

# To run:
1. Run PG-stockmod.ipynb for policy gradient based RL algorithm or Q-stockmod.ipynb for DQN RL algorithm.

# Overview of the files:
market_env.py - initializes RL training enviroment in which agent acts
market_model_builder.py - creates CNN architecture for task optimization
model_builder.py - returns saved model
Q-stockmod.ipynb - runs DQN model
PG-stockmod.ipynb - runs PG model
