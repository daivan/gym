from custom_envs.SmallerScreen import SmallerScreen
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env

env = SmallerScreen()

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

check_env(env, warn=True)

plt.imshow(observation)
plt.show()

