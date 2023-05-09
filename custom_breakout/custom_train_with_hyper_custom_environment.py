import gymnasium as gym
import os
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import numpy as np
from custom_breakout_only_ball_normal_score import CustomBreakoutOnlyBallNormalScore

# model
algo = 'ppo'
policy = 'CnnPolicy'
n_steps = 128
n_epochs = 4
batch_size = 256
n_timesteps = 1e6
learning_rate_initial = 2.5e-4
clip_range_initial = 0.1
vf_coef = 0.5
ent_coef = 0.01

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = CustomBreakoutOnlyBallNormalScore()

from typing import Callable, Union
def linear_scedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress: float) -> float:
        return progress * initial_value
    
    return func

learning_rate_schedule = linear_scedule(learning_rate_initial)
clip_range_schedule = linear_scedule(clip_range_initial)


seed = 123
model = PPO(policy='MlpPolicy', 
            env = env,
            n_steps         = n_steps,
            n_epochs        = n_epochs,
            batch_size      = batch_size,
            learning_rate   = learning_rate_schedule,
            clip_range      = clip_range_schedule,
            vf_coef         = vf_coef,
            ent_coef        = ent_coef,
            verbose         = 1,
            tensorboard_log = log_dir)


TIMESTEPS = 10000

for i in range(1, 300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ppo")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


