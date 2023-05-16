import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class SmallerScreen(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        self.env = gym.make("ALE/FishingDerby-v5", obs_type="grayscale")
        self.observation_space = self.env.observation_space
        #layered_observation = np.dstack((self.env.observation_space, self.env.observation_space))
        #self.observation_space = layered_observation
        self.observation_space = Box(low=0, high=255, shape=(65, 80, 1), dtype=np.uint8)
        self.action_space = self.env.action_space

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)

        observation = observation[70:135,10:90]
        reshaped_array = np.expand_dims(observation, axis=-1)
        # Modify the reward or the state transition here
        return reshaped_array, reward, done, truncated, info
    
    def reset(self):
        self.env.reset()
        observation = np.random.randint(0, 256, size=(65, 80, 1), dtype=np.uint8)
        into = {'empty': 5}
        return observation, into