import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("ALE/Breakout-v5", obs_type="grayscale")

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

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
first_observation = observation
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
second_observation = observation
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
third_observation = observation

layered_observation = np.dstack((first_observation, second_observation, third_observation))
print(layered_observation.shape)
#plt.imshow(third_observation)
#plt.show()


# assume layered_observation has shape (n_rows, n_cols, 2)
squashed_observation = np.concatenate((first_observation, second_observation), axis=1)

plt.imshow(squashed_observation)
plt.show()

