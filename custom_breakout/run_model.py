from stable_baselines3 import *
from custom_breakout_only_ball import CustomBreakoutOnlyBall

algorithm = "ppo"

# Change these two to match your model and file
models_dir = "models/" + algorithm
model_file = "920000"

# change how many times it will play
episodes = 10

model_path = f"{models_dir}/{model_file}"

env = CustomBreakoutOnlyBall()

model = PPO.load(model_path, env=env)


for episode in range(episodes):
    observation = env.reset()
    observation = observation[0]
    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)

env.close()