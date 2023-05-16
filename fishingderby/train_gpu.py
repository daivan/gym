import gymnasium as gym
import os
from stable_baselines3 import *
import sys
import torch
import numpy as np
from gymnasium.spaces import Box

algorithm = "ppo"
if len(sys.argv) > 1:
    accepted_algorithm_array = ["ppo", "a2c", "dqn"]
    input_algorithm = sys.argv[1]
    if input_algorithm in accepted_algorithm_array:
        algorithm = input_algorithm
    else:
        print('Invalid algorithm. Accepted algorithms are: ' + str(accepted_algorithm_array))

print('algorithm: ' + algorithm)

models_dir = "models/" + algorithm
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)



class CustomEnv(gym.Env):
    def __init__(self, env_name):
        self.env = gym.make(env_name, obs_type="grayscale")
        #self.observation_space = self.env.observation_space
        self.observation_space = Box(low=0, high=1, shape=(210, 160), dtype=np.uint8)
        self.action_space = self.env.action_space
        #self.first_observation = np.zeros((210, 160), dtype=np.uint8)
        #self.second_observation = np.zeros((210, 160), dtype=np.uint8)

    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)
        #observation = observation[50:195,5:155]
        observation[observation > 0] = 1
        # Modify the reward or the state transition here
        # TODO: Implement your custom logic here
        #gray_img = np.mean(observation, axis=-1)
        #layered_observation = np.dstack((self.first_observation, self.second_observation, gray_img))

        #self.first_observation = self.second_observation

        #self.second_observation = gray_img

        # Modify the reward or the state transition here
        #return layered_observation, reward, done, truncated, info
    
        # We only play with first life.
        if info['lives'] < 5:
            done = True

        return observation, reward, done, truncated, info
    
    def reset(self):
        return self.env.reset()
    

env = CustomEnv('ALE/Breakout-v5')

num_gpus = torch.cuda.device_count()
#device = f'cuda:{torch.cuda.current_device()}'

device = 'cpu'
env.reset()
if algorithm == "ppo":
    model = PPO('CnnPolicy', env, device=device, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001)
elif algorithm == "a2c":
    model = A2C("CnnPolicy", env, device=device, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001)
elif algorithm == "dqn":
    model = DQN("CnnPolicy", env, device=device, verbose=1, batch_size=16, tensorboard_log=log_dir)

TIMESTEPS = 10000

step = 0
while True:
    step = step + 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm+"_the_real_cpu")
    model.save(f"{models_dir}/{TIMESTEPS*step}")
