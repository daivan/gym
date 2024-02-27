import os
from stable_baselines3 import *
import sys
from environments.WoodBlockPuzzle import WoodBlockPuzzle

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


env = WoodBlockPuzzle()

env.reset()
if algorithm == "ppo":
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
elif algorithm == "a2c":
    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
elif algorithm == "dqn":
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

step = 0
while True:
    step = step + 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm)
    model.save(f"{models_dir}/{TIMESTEPS*step}")
