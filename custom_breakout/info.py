import gymnasium as gym
import os
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env
#from custom_breakout_only_ball import CustomBreakoutOnlyBall
from custom_breakout import CustomBreakout

#env = CustomBreakoutOnlyBall()
env = CustomBreakout()

env.reset()

print('-------- Observation Space ---------')
print(env.observation_space)
print('-------- Action Space ---------')
print(env.action_space)
print(env.action_space.sample())

print('-------- Step ---------')
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print('Observation: ', observation)
print('Reward: ', reward)
print('Terminated: ', terminated)
print('Truncated: ', truncated)
print('Info: ', info)
print(observation.shape)
observation[observation > 0] = 1


shrink = observation[189:193,8:152]

board_indices = np.where(shrink == 1)

# Print the indices
if board_indices[1].size == 0:
    #print("No board")
    board_location = 0
    pass
else:
    board_location = board_indices[1][0]
    pass

print(board_indices)
print(board_location)
plt.imshow(shrink)
plt.show()

#check_env(env)