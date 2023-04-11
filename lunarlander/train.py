import gymnasium as gym
import os

# Check what descrete it is https://stable-baselines.readthedocs.io/en/master/guide/algos.html to see what algorithm to use
from stable_baselines3 import PPO

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make('LunarLander-v2')

env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(1, 300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")



#print("sample step: ", env.action_space.sample())
#print("sample observation shape:", env.observation_space.shape)
#print("sample observation sample:", env.observation_space.sample())

env.close()