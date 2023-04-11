import gymnasium as gym
import os

# Check what descrete it is https://stable-baselines.readthedocs.io/en/master/guide/algos.html to see what algorithm to use
from stable_baselines3 import PPO

# Change these two to match your model and file
models_dir = "models/ppo"
model_file = "130000"

# change how many times it will play
episodes = 10

model_path = f"{models_dir}/{model_file}"

env = gym.make('CartPole-v1', render_mode="human")

model = PPO.load(model_path, env=env)

for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)

    
env.close()