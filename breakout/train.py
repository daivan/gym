import gymnasium as gym
import os
from stable_baselines3 import *
import sys
from custom_breakout_with_stacking_images import CustomBreakoutWithStackingImages

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


env = gym.make("ALE/Breakout-v5")
#env = CustomBreakoutWithStackingImages()

env.reset()
if algorithm == "ppo":
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001)
elif algorithm == "a2c":
    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001)
elif algorithm == "dqn":
    model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001)

TIMESTEPS = 10000

step = 0
while True:
    step = step + 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm+"_cpu")
    model.save(f"{models_dir}/{TIMESTEPS*step}")
