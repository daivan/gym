import gymnasium as gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import os

#env = gym.make('ALE/Breakout-v5', render_mode='human')
#env = gym.make('ALE/Breakout-v5')
#env = EpisodicLifeEnv(env)

env=make_vec_env(env_id='ALE/Breakout-v5', 
                 n_envs=8,
                 wrapper_class=EpisodicLifeEnv, 
                 )

algorithm = "ppo"
models_dir = "models/" + algorithm
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


seed = 123
model = PPO(policy='CnnPolicy', 
            env = env,
            verbose         = 1,
            seed            = seed,
            tensorboard_log = log_dir)

model.learn(total_timesteps = 1e7,
            tb_log_name     ='2.1_envs')
model.save(f"{models_dir}/last_model")
'''
observation = env.reset()


for step in range(int(1e3)):

    action = model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)

    print('reward: ' + str(reward))
    print('done: ' + str(done))

    time.sleep(0.1)

    if done:
        print('final reward: ' + str(reward))
        break
        env.reset()
'''
env.close()