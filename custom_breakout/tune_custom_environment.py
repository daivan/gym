import gymnasium as gym
import os
import numpy as np
from matplotlib import pyplot as plt
from custom_breakout_only_ball import CustomBreakoutOnlyBall
from stable_baselines3.common.env_checker import check_env

env = CustomBreakoutOnlyBall()

env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)


#print(env)
#print(observation)

while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    #print(reward)
    if terminated:
        env.reset()
