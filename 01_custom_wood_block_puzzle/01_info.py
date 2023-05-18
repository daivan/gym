from environments.WoodBlockPuzzle import WoodBlockPuzzle
from stable_baselines3.common.env_checker import check_env

env = WoodBlockPuzzle()

env.reset()

print('-------- Observation Space ---------')
print(env.observation_space)
print('-------- Action Space ---------')
print(env.action_space)
print(env.action_space.sample())

print('-------- Step ---------')
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print('Random action: ', action)
print('Observation: ', observation)
print('Reward: ', reward)
print('Terminated: ', terminated)
print('Truncated: ', truncated)
print('Info: ', info)


check_env(env, warn=True)