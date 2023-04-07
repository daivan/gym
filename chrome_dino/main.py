# MSS used for screen cap
from mss import mss
# Sending command
import pydirectinput
# allows for frame processing
import cv2
# Transformational framwork
import numpy as np
# OCR for game over extraction
import pytesseract
# Visualize capture frames
from matplotlib import pyplot as plt
# Brin in time for pauses
import time
# environment components
from gym import Env
# Box = image
# Discrete = action
from gym.spaces import Box, Discrete


import os
# import base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# check environment
from stable_baselines3.common import env_checker
# Import the DQN algorithm
from stable_baselines3 import DQN


class WebGame(Env):
    
    def __init__(self):
        super().__init__()

        # shape = 1 = image, 83 = height, 100 = width # You could send multiple images
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        # We define 3 actions 0,1,2 with Descrete(3)
        self.action_space = Discrete(3)
        
        # Define extration parameters for the game
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 540, 'left': 920, 'width': 70, 'height': 70}

    def step(self, action):
        # Action key - 0 = Space, 1 = Duck(down), 2 = No action
        action_map = {
            0: 'space', 
            1: 'down', 
            2: 'no_op'
        }

        if action != 2:
            pydirectinput.press(action_map[action])

        # check if game is done
        done = self.get_done()

        # ge tthe next observation
        new_observation = self.get_observation()

        # get the reward. We get a point for every frame we are alive
        reward = 1

        # Stable baselines requires a dict for info
        info = {}

        return new_observation, reward, done, info

    # Visualize the game
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()


    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()

    def close(self):
        cv2.destroyAllWindows()


    def get_observation(self):

       
        # screen grab
        raw = self.cap.grab(self.game_location)

        # convert to array
        raw = np.array(raw)

        # If the datatype was wrong you could run, you can run the following to get check the datatype
        # print(raw.dtype)
        # raw = raw.astype(np.uint8)
        
        # extract first three channels
        #raw = raw[:,:,3]

        # grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Resize to the correct size
        resized = cv2.resize(gray, (100, 83))

        # Add channels first
        channel = np.reshape(resized, (1, 83, 100))

        return channel

        

    def get_done(self) -> bool:
        # Get done
        done_cap = self.cap.grab(self.done_location)

        done_cap = np.array(done_cap)

        # Get the correct value for the done
        current_summary = np.sum(done_cap)

        # Correct value for the done
        # This has been precalculated using np.sum
        correct_game_over_summary = 2997096
        
        # Check if the two images are the same
        
        if current_summary == correct_game_over_summary:
            return True
        return False
   

# Same as always
class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'model_{}'.format(self.n_calls))
            self.model.save(model_path)

env = WebGame()
obs = env.get_observation()

# Play 10 games
#for episode in range(10):
#    obs = env.reset()
#    done = False
#    total_reward = 0

#    while not done:
#        obs, reward, done, info = env.step(env.action_space.sample())
#        total_reward += reward

#    print('Episode: {}, Total Reward: {}'.format(episode, total_reward))

env_checker.check_env(env)

callback = TrainAndLoggingCallback(check_freq=300, save_path='./train/')

model = DQN('CnnPolicy', 
            env, 
            verbose=1, 
            tensorboard_log='./logs/', 
            buffer_size=1200000, 
            learning_starts=0
            )

model.learn(total_timesteps=5000, callback=callback)

# Load the model first
# train = folder
# best_model_88000 = zip file
#model.load(os.path.join('train', 'best_model_88000'))
# Play 1 game
#for episode in range(1):
#    obs = env.reset()
#    done = False
#    total_reward = 0

#    while not done:
#        action, _ = model.predict(obs)
#        obs, reward, done, info = env.step(int(action))
#        time.sleep(0.01)
#        total_reward += reward
#
#    print('Episode: {}, Total Reward: {}'.format(episode, total_reward))
#    time.sleep(2)
