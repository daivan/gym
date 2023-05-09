import gymnasium as gym
import os
from stable_baselines3 import PPO
import optuna
import numpy as np
from optuna.integration.tensorboard import TensorBoardCallback

models_dir = "models/ppo"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def optimize_ppo(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 1, 100),
    }

def create_acrobot_model(hyperparameters):
    env = gym.make("Acrobot-v1")
    model = PPO("MlpPolicy", env, verbose=0, **hyperparameters)
    return model, env

def objective(trial):
    hyperparameters = optimize_ppo(trial)
    model, env = create_acrobot_model(hyperparameters)
    model.learn(10000)
    rewards = []
    for i in range(100):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return np.mean(rewards)

study = optuna.create_study(direction="maximize")
callback = TensorBoardCallback("logs", metric_name="rewards")
study.optimize(objective, n_trials=100, callbacks=[callback])

print(study.best_params)