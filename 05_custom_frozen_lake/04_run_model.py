from frozen_lake_env import FrozenLakeEnv
from stable_baselines3 import *

algorithm = "ppo"

# Change these two to match your model and file
models_dir = "models/" + algorithm
model_file = "1870000"

# change how many times it will play
episodes = 10

model_path = f"{models_dir}/{model_file}"


env = FrozenLakeEnv('human')

if algorithm == "ppo":
    model = PPO.load(model_path, env=env)
elif algorithm == "a2c":
    model = A2C.load(model_path, env=env)
elif algorithm == "dqn":
    model = DQN.load(model_path, env=env)



for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action, _ = model.predict(observation)
        action = int(action)
        observation, reward, done, truncated, info = env.step(action)

env.close()