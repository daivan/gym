import gymnasium as gym

env = gym.make("ALE/FishingDerby-v5", obs_type="grayscale")

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



