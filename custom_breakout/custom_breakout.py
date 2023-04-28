import gymnasium as gym
import numpy as np
from gymnasium.spaces import *
import pygame

class CustomBreakout(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        #pygame.init()
        #self.screen = pygame.display.set_mode((210, 160))

        super().__init__()
        #self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode='human')
        self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
        self.observation_space = self.env.observation_space
        #layered_observation = np.dstack((self.env.observation_space, self.env.observation_space))
        #self.observation_space = layered_observation

        self.observation_space = self.env.observation_space
        # Live
        #self.observation_space = MultiDiscrete([210, 210])
        self.action_space = self.env.action_space

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)

        return observation, reward, done, truncated, info

    
    def reset(self):
        self.env.reset()
        empty_observation = np.zeros(self.observation_space.shape, dtype=np.int8)
        # create a numpy array of zeros
        #empty_observation = np.zeros((210), dtype=np.uint8)

        into = {'empty': 5}
        return empty_observation, into
    
    def render(self):
        #return self.env.render(mode=mode)

        #self.screen.fill((0, 0, 0))
        #pygame.draw.rect(self.screen, (255, 0, 0), (100, 100, 50, 50))
        #pygame.surfarray.make_surface(self.current_observation)
        #image_surface = pygame.surfarray.make_surface(self.current_observation)
        #self.screen.blit(image_surface, (0, 0))

        #pygame.display.flip()
        pass