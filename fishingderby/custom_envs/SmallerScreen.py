import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

#
# We halve the screen and only focus on our fish
#
class SmallerScreen(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode='none'):
        super().__init__()
        if (render_mode =='none'):
            self.env = gym.make("ALE/FishingDerby-v5")
        elif (render_mode == 'human'):
            self.env = gym.make("ALE/FishingDerby-v5", render_mode='human')

        self.observation_space = self.env.observation_space

        self.observation_space = Box(low=0, high=255, shape=(65, 80, 3), dtype=np.uint8)
        self.action_space = self.env.action_space

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)

        observation = observation[70:135,10:90]
        # Modify the reward or the state transition here
        return observation, reward, done, truncated, info
    
    def reset(self):
        self.env.reset()
        observation = np.random.randint(0, 256, size=(65, 80, 3), dtype=np.uint8)
        into = {'empty': 5}
        return observation, into