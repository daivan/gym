import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
import pygame

'''

This is to test out stages
You get to pick 3 actions in the first stage (rock paper scissor)
You get to pick 10 actions in the second stage (rock, rock, paper, paper, scissor, scissor, rock, paper, scissor, rock) Random.
You get points based on the actions you picked in the second stage.
'''
class RockPaperScissor(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "weapons": spaces.Box(0, 2, shape=(3,), dtype=int),
                'enemies': spaces.Box(0, 2, shape=(10,), dtype=int) # can only be 0 as of now
            }
        )

        # Define the action spaces
        # 0-2: Rock, Paper, Scissor
        # 0-9: opponents to face
        self.action_space = spaces.Discrete(29) #30 alternatives 0-9 = rock+opponents, 10-19 = paper+opponents, 20-29 = scissor+opponents

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "weapons": self.weapons,
            "enemies": self.enemies
            }
    
    def _get_info(self):
        return {"nothing matters": "not important"}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the number of turns
        self.turns = 0

        self.weapons = np.random.randint(low=0, high=3, size=3) # is the same as, but random values np.array([0, 1, 2])
        self.enemies = np.random.randint(low=0, high=3, size=10) # is the same as, but random values np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):

        #first_action = action[0]

        observation = self._get_obs()
        info = self._get_info()

        self.turns += 1

        isGameOver = self.is_game_over()
        if isGameOver == True:
            reward = 0
            terminated = True
            return observation, reward, terminated, False, info

        reward = self.count_reward(action)

        terminated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        box_count = 0
        # First we draw the target
        for box_frame in self._fill_box:
            if box_frame == 1:
                vertical, horizontal = self.getVerticalAndHorizontal(box_count)
                pygame.draw.rect(
                    canvas,
                    (255, 0, 0),
                    pygame.Rect(
                        ((
                    pix_square_size * horizontal), # horizontal
                    (pix_square_size * vertical) # vertical
                    ),
                        (pix_square_size, pix_square_size),
                    ),
                )
            box_count += 1
        
    

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def getVerticalAndHorizontal(self, box_count):
        # get the vertical and horizontal from a 8x8 grid
        vertical, horizontal = divmod(box_count, 8)
        return vertical, horizontal
    
    def is_game_over(self):

        is_game_over = False

        if self.turns >= 100:
            is_game_over = True

        return is_game_over


    def can_place_block(self, action):

        vertical, horizontal = self.getVerticalAndHorizontal(action)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        #self._fill_box[row * 8 + col] = 1

        can_place_block = True
        if self._fill_box[horizontal * 8 + vertical] == 1:
            can_place_block = False
            
        current_row = horizontal
        current_col = vertical
        # Explore the adjacent tiles
        for direction in directions:
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
  
            if 7 < new_row or new_row < 0:
                can_place_block = False
                continue

            if 7 < new_col or new_col < 0:
                can_place_block = False
                continue

            if self._fill_box[new_row * 8 + new_col] == 1:
                can_place_block = False

        return can_place_block
    
    def place_block(self, action):
        #self._fill_box[action] = 1
        vertical, horizontal = self.getVerticalAndHorizontal(action)
        self.fill_adjacent_tiles(horizontal, vertical)
        return True    
    
    def fill_adjacent_tiles(self, row, col):

        # Define the adjacent positions (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Mark the current tile as 1
        self._fill_box[row * 8 + col] = 1

        current_row = row
        current_col = col
        # Explore the adjacent tiles
        for direction in directions:
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                self._fill_box[new_row * 8 + new_col] = 1


    def count_reward(self, action):

        #there are 30 actions

        # If action between 0-9, then picked first alternative
        if action < 10:
            weapon = self.weapons[0]
            enemy = self.enemies[action]
            pass
        # If action between 10-19, then picked second alternative
        elif action < 20:
            weapon = self.weapons[1]
            enemy = self.enemies[action-10]
            pass
        # If action between 20-29, then picked third alternative
        elif action < 30:
            weapon = self.weapons[2]
            enemy = self.enemies[action-20]
            pass

        # 0 = rock
        # 1 = paper
        # 2 = scissor

        # Its a draw, reward = 0
        if weapon == enemy:
            return 0
        
        # You win, reward = 3
        if (weapon == 0 and enemy == 2) or (weapon == 1 and enemy == 0) or (weapon == 2 and enemy == 1):
            return 3
        
        # If you lose, reward = -10
        if (weapon == 0 and enemy == 1) or (weapon == 1 and enemy == 2) or (weapon == 2 and enemy == 0):
            return -10
        
        