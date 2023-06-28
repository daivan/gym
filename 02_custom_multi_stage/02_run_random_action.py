from environments.RockPaperScissor import RockPaperScissor

#env = WoodBlockPuzzle('human')
env = RockPaperScissor()

episodes = 10
max_reward = 0
for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    
    temporary_reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        temporary_reward = temporary_reward + reward
    
    if max_reward < temporary_reward:
        max_reward = temporary_reward

print('Max reward: ', max_reward)
env.close()
