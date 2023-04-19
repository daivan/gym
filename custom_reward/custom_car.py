import gymnasium as gym
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Check what descrete it is https://stable-baselines.readthedocs.io/en/master/guide/algos.html to see what algorithm to use
from stable_baselines3 import PPO

#env = gym.make('CarRacing-v2', render_mode="human")

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class CustomEnv(gym.Env):
    def __init__(self, env_name):
        #self.env = gym.make(env_name, render_mode="human")
        self.env = gym.make(env_name)
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

        return observation, reward, done, truncated, info
    
    def reset(self):
        return self.env.reset()
    

env = CustomEnv('CarRacing-v2')

# change how many times it will play
#episodes = 10

#for episode in range(episodes):
#    observation = env.reset()
#    observation = observation[0]
#    done = False
#    score_state = 0
#    while not done:
#        action = env.action_space.sample()
#        observation, reward, done, truncated, info = env.step(action)
#        score_state = score_state + reward
#        print(score_state)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(1, 300):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
