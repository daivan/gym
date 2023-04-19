import gymnasium as gym
import os

# Check what descrete it is https://stable-baselines.readthedocs.io/en/master/guide/algos.html to see what algorithm to use
from stable_baselines3 import PPO

# Change these two to match your model and file
models_dir = "models/ppo"
model_file = "430000"

# change how many times it will play
episodes = 10

model_path = f"{models_dir}/{model_file}"


class CustomEnv(gym.Env):
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="human")
        #self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.number_of_minus_in_a_row = 0
        self.max_number_of_minus_in_a_row = 50
        self.total_reward = 0
    def step(self, action):
        # Call the original step method
        observation, reward, done, truncated, info = self.env.step(action)
        # Modify the reward or the state transition here
        # TODO: Implement your custom logic here
        self.total_reward += reward
        if self.total_reward < -10:
            done = True
        #if reward < 0:
        #    self.number_of_minus_in_a_row += 1
        #else:
        #    self.number_of_minus_in_a_row = 0

        #if self.number_of_minus_in_a_row >= self.max_number_of_minus_in_a_row:
        #    done = True

        return observation, reward, done, truncated, info
    
    def reset(self):
        return self.env.reset()
    

env = CustomEnv('CarRacing-v2')


model = PPO.load(model_path, env=env)

for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)

    
env.close()