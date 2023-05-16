import gymnasium as gym
import os
from stable_baselines3 import *
import sys

algorithm = "dqn"
if len(sys.argv) > 1:
    accepted_algorithm_array = ["ppo", "a2c", "dqn"]
    input_algorithm = sys.argv[1]
    if input_algorithm in accepted_algorithm_array:
        algorithm = input_algorithm
    else:
        print('Invalid algorithm. Accepted algorithms are: ' + str(accepted_algorithm_array))

print('algorithm: ' + algorithm)


# Change these two to match your model and file
models_dir = "models/" + algorithm
model_file = "4210000"

model_path = f"{models_dir}/{model_file}"

env = gym.make("ALE/Breakout-ram-v5")

learning_rate = 0.000001
custom_objects = { 'learning_rate': learning_rate }

env.reset()
if algorithm == "ppo":
    model = PPO.load(model_path, env=env, custom_objects=custom_objects)
elif algorithm == "a2c":
    model = A2C.load(model_path, env=env, custom_objects=custom_objects)
elif algorithm == "dqn":
    model = DQN.load(model_path, env=env, custom_objects=custom_objects)

TIMESTEPS = 10000

step = 421
while True:
    step = step + 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm+"_custom_learning_rate_2")
    model.save(f"{models_dir}/{TIMESTEPS*step}")
