import gymnasium as gym
import os
from stable_baselines3 import *
import numpy as np
from custom_breakout_with_stacking_images import CustomBreakoutWithStackingImages

models_dir = "models/custom_a2c"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = CustomBreakoutWithStackingImages()

model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)


TIMESTEPS = 10000

for i in range(1, 300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="custom_A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


