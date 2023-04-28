import gymnasium as gym
import os
import numpy as np
from matplotlib import pyplot as plt
from custom_breakout_with_stacking_images import CustomBreakoutWithStackingImages

env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
#env = gym.make("ALE/Breakout-v5")

#env = CustomBreakoutWithStackingImages()

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

print(info['lives'])
#gray_img = np.mean(observation, axis=-1)
observation
new_image = observation[50:195,5:155]
#new_image = new_image[-50:,:]
#resized_arr = np.resize(observation, (105, 80))
print(new_image.shape)

new_image[new_image > 0] = 1

plt.imshow(new_image)
plt.show()


