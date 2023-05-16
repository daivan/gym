import gymnasium as gym

env = gym.make("ALE/FishingDerby-v5", render_mode="human")

env.reset()

episodes = 10
for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

env.close()
