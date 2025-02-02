import random
import time
import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces


WIDTH, HEIGHT = 400 , 400
ROW, COLUMN = 10, 10
FPS = 10
BOX_WIDTH, BOX_HEIGHT = 40, 40

score = 0
previousScore = 0
heroPosition = [5, 4]
goalPosition = [3, 3]
enemyPosition = [2, 2]
gameOver = False

def collision(heroPosition, goalPosition, enemyPosition, score, gameOver):
  
    if heroPosition == goalPosition:
        score += 1
        goalPosition = [random.randrange(1, COLUMN), random.randrange(1, ROW)]
    
    gameOver = False
    if heroPosition == enemyPosition:
        gameOver = True

    # Collision with top wall
    if heroPosition[1] < 0:
        gameOver = True

    # Collision with bottom wall
    if heroPosition[1] > ROW - 1:
        gameOver = True
    
    # Collision with left wall
    if heroPosition[0] < 0:
        gameOver = True
    
    # Collision with right wall
    if heroPosition[0] > COLUMN - 1:
        gameOver = True

    return goalPosition, score, gameOver



class FrozenLakeEnv(gym.Env):

    def __init__(self, render_mode = None):
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=40, shape=(7, ), dtype=np.int64)


    def step(self, action):

        self.goalPosition, self.score, self.gameOver = collision(self.heroPosition, self.goalPosition, self.enemyPosition, self.score, self.gameOver)

        if action == 0:
            self.heroPositionX = self.heroPosition[0]
            self.heroPositionY = self.heroPosition[1] - 1
        if action == 1:
            self.heroPositionX = self.heroPosition[0]
            self.heroPositionY = self.heroPosition[1] + 1
        if action == 2:
            self.heroPositionX = self.heroPosition[0] + 1
            self.heroPositionY = self.heroPosition[1]
        if action == 3:
            self.heroPositionX = self.heroPosition[0] - 1
            self.heroPositionY = self.heroPosition[1]            
        self.heroPosition = [self.heroPositionX, self.heroPositionY]
                    

        self.observation = [
            self.heroPosition[0],
            self.heroPosition[1],
            self.goalPosition[0],
            self.goalPosition[1],
            self.enemyPosition[0],
            self.enemyPosition[1],
            action]
                            
        self.observation = np.array(self.observation)

        if self.gameOver:
            self.done = True


        # reward
        # A: death punishment
        reward_a = 0
        if self.done:
            reward_a = -100
        
        
        # B: eating apple
        reward_b = 0
        if self.previousScore < self.score:
            reward_b = 10
            self.previousScore = self.score

        # C: moving closer to apple
        try:
            self.distance = abs(self.heroPosition[0] - self.goalPosition[0]) + abs(self.heroPosition[1] - self.goalPosition[1])
        except Exception as e:
            print(f"An error occurred: {e}")
        
        if self.distance > self.previousDistance:            
            reward_c = -1
        elif self.distance < self.previousDistance:
            reward_c = 1
        else:
            reward_c = 0
        self.previousDistance = self.distance


        self.reward = reward_a + reward_b + reward_c

        self.info = {}

        if self.render_mode == 'human':
            self.render()

        terminated = False

        return self.observation, self.reward, self.done, terminated, self.info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.score = 0
        self.previousScore = 0
        self.heroPosition = [5, 4]
        self.goalPosition = [3, 3]
        self.enemyPosition = [2, 2]
        self.gameOver = False
        self.done = False
        self.distance = 0
        self.previousDistance = 0

        self.observation = [
            self.heroPosition[0],
            self.heroPosition[1],
            self.goalPosition[0],
            self.goalPosition[1],
            self.enemyPosition[0],
            self.enemyPosition[1],            
            0]
        self.observation = np.array(self.observation)

    
    
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption('Snake Game')
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial_bold', 380)
            self.render()

        info = {}
        return self.observation, info
    
    def render(self, render_mode='human'):
        # draw
        # Draw background
        self.display.fill((67, 70, 75))

        # Draw score
        img = self.font.render(str(score), True, (57, 60, 65))
        self.display.blit(img, (WIDTH // 2 - 100, HEIGHT // 2 - 100))

        # Draw Hero
        pygame.draw.rect(self.display, 'BLUE', (self.heroPosition[0] * 40, self.heroPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

        # Draw Goal
        pygame.draw.rect(self.display, 'GREEN', (self.goalPosition[0] * 40, self.goalPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

        # Draw Enemy
        pygame.draw.rect(self.display, 'RED', (self.enemyPosition[0] * 40, self.enemyPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

        pygame.display.update()

        # optional
        if self.done:
            time.sleep(0.5)


    def close(self):
        pygame.quit()