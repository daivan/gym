import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class CustomBreakoutWithStackingImages(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
        self.observation_space = self.env.observation_space
        #layered_observation = np.dstack((self.env.observation_space, self.env.observation_space))
        #self.observation_space = layered_observation
        self.observation_space = Box(low=0, high=255, shape=(145, 150, 3), dtype=np.uint8)
        self.action_space = self.env.action_space
        self.first_observation = np.zeros((145, 150), dtype=np.uint8)
        self.second_observation = np.zeros((145, 150), dtype=np.uint8)

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)
        #observation[observation > 0] = 1
        observation = observation[50:195,5:155]
        layered_observation = np.dstack((self.first_observation, self.second_observation, observation))
        
        self.first_observation = self.second_observation

        self.second_observation = observation

        # We only play with first life.
        if info['lives'] < 5:
            done = True

        # Modify the reward or the state transition here
        return layered_observation, reward, done, truncated, info
    
    def reset(self):
        self.env.reset()
        empty_observation = np.zeros((145, 150), dtype=np.uint8)
        layered_observation = np.dstack((empty_observation, empty_observation, empty_observation))
        into = {'empty': 5}
        return layered_observation, into