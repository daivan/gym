import gymnasium as gym
import os
from stable_baselines3 import *
import numpy as np
from custom_breakout_only_ball import CustomBreakoutOnlyBall

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = CustomBreakoutOnlyBall()

episodes = 10

for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action = env.action_space.sample()
        #action = 1
        observation, reward, done, truncated, info = env.step(action)

    
env.close()

