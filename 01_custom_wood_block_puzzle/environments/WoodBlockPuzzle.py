import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
import pygame


class WoodBlockPuzzle(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(63)

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
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        # self._fill_box = [0,0,1,0 + ..  64]
        self._fill_box = self.np_random.integers(0, 2, size=(64,), dtype=int)
        self._next_block = 0 # 0 can only be 0

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):

        observation = self._get_obs()
        info = self._get_info()

        isGameOver = self.is_game_over(action)
        if isGameOver == True:
            reward = 0
            terminated = True
            return observation, reward, terminated, False, info
        

        canPlaceBlock = self.can_place_block(action)
        if canPlaceBlock == False:
            reward = -1
            terminated = False
            return observation, reward, terminated, False, info
        

        self.place_block(action)

        reward = self.count_reward()

        # An episode is done iff the agent has reached the target
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
    
    def is_game_over(self, action):

        if self._next_block == 0:
            pass
        return False


    def can_place_block(self, action):

        return True
    
    def place_block(self, action):
        #self._fill_box[action] = 1
        vertical, horizontal = self.getVerticalAndHorizontal(action)
        self.fill_adjacent_tiles(horizontal, vertical)
        return True    
    
    def fill_adjacent_tiles(self, row, col):
        print(row, col)
        # Define the adjacent positions (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        #directions = [(-1, 0)]

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
        #    # Check if the new position is within the self._fill_box boundaries
        #    if 0 <= new_row < 8 and 0 <= new_col < 8:
        #        # Check if the tile is not already marked as 1
        #        if self._fill_box[new_row * 8 + new_col] != 1:
        #            # Mark the tile as 1 and enqueue it for further exploration
        #            self._fill_box[new_row * 8 + new_col] = 1
        #            queue.append((new_row, new_col))

    ### Example usage
    #board = [0] * 64

    ### Place a block at row 3, column 4
    #block_row = 3
    #block_col = 4
    #fill_adjacent_tiles(board, block_row, block_col)

    def count_reward(self):

        reward = 0
        board = self._fill_box
        for i in range(8):
            # Check if the row is full of 1s
            if all(board[i*8 + j] == 1 for j in range(8)):
                print("Resetting Row", i, "to 0.")
                for j in range(8):
                    self._fill_box[i*8 + j] = 0
                reward += 1

            # Check if the column is full of 1s
            if all(board[j*8 + i] == 1 for j in range(8)):
                print("Resetting Column", i, "to 0.")
                for j in range(8):
                    self._fill_box[j*8 + i] = 0
                reward += 1

        return reward