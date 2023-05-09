import gymnasium as gym
import numpy as np
from gymnasium.spaces import *
import pygame

class CustomBreakoutOnlyBallNormalScore(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        #pygame.init()
        #self.screen = pygame.display.set_mode((210, 160))

        super().__init__()
        #self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale", render_mode="human")
        self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale")
        self.observation_space = self.env.observation_space
        #layered_observation = np.dstack((self.env.observation_space, self.env.observation_space))
        #self.observation_space = layered_observation

        # Live
        self.observation_space = MultiDiscrete([210, 210])
        self.action_space = self.env.action_space

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action) 
        
        observation[observation > 0] = 1
        ball = observation[93:188,8:152]

        board = observation[189:193,8:152]

        #empty_observation = np.zeros(self.observation_space.shape, dtype=np.int8)
        # Modify the reward or the state transition here
        #return empty_observation, reward, done, truncated, info

        board_indices = np.where(board == 1)
        ball_indices = np.where(ball == 1)
        if ball_indices[1].size == 0:
            #print("No ball")
            ball_location = 0
            self.step(1)
        else:
            ball_location = ball_indices[1][0]
            pass

    
        board_location = board_indices[1][0]
 
        board_and_ball_observation = np.array([board_location, ball_location])

        if info['lives'] < 5:
            done = True
            return board_and_ball_observation, reward, done, truncated, info
           
        # Chop the image
        return board_and_ball_observation, reward, done, truncated, info

    
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