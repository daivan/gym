import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

models_dir = "models/ppo"
log_dir = "logs"

def optimizePpo(trail):
    return {
        # 'n_steps': 8192 * 2,
        # 'n_steps': trail.suggest_int('n_steps', 4096*2, 16384, step=64),
        'n_steps': trail.suggest_int('n_steps', 64, 4096, step=64),
        'gamma': trail.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trail.suggest_loguniform('learning_rate', 3e-5, 3e-4),
        'clip_range': trail.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trail.suggest_uniform('gae_lambda', 0.8, 0.9999),
        'ent_coef': trail.suggest_loguniform('ent_coef', 1e-4, 0.1),
    }

TIMESTEP = 4096 * 5

def optimizeModel(trail):
    try:
        model_params = optimizePpo(trail)

        env = gym.make("Acrobot-v1")

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **model_params)

        model.learn(total_timesteps=TIMESTEP, tb_log_name="MAY_OPTUNA")

        # play 5 games as evaluation, to se how much reward it gains
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

        model.save(f"logs/may_model-{trail.number}")

        return mean_reward

    except Exception as e:
        print(e)
        return -1000

study = optuna.create_study(direction='maximize')
study.optimize(optimizeModel, n_trials=100, n_jobs=10)

print(study.best_trial)
print(study.best_params)