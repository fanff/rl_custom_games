import logging

import click
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MFLow


@click.command()
@click.option("--model_idx", default="logs/default")
@click.option("--n_envs", default=8, type=int, show_default=True)
@click.option("--experiment_name", default="lk")
@click.option("--parallel_env", default=False, type=bool)
@click.option("--logfreq", default=1000, type=int)
@click.option("--from_scratch", default=False, type=bool)
def train_ttris(model_idx, n_envs, experiment_name, parallel_env, logfreq, from_scratch):
    brick_set = "traditional"
    board_height = 12
    board_width = 6
    max_step = 2000

    logging.basicConfig(level=logging.INFO)

    # env = make_vec_env(CustomTetris, n_envs=n_envs)
    logger = logging.getLogger("ttrain")

    logger.info("model_idx: %s, n_envs: %s, experiment_name: %s, parallel_env: %s, logfreq: %s, from_scratch: %s",
                model_idx, n_envs, experiment_name, parallel_env, logfreq, from_scratch)
    if parallel_env:
        env = SubprocVecEnv([CustomTetris] * n_envs)
    else:
        env = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)

    evalenv = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
    eval_callback = EvalCallback(evalenv, best_model_save_path=f"./{model_idx}/",
                                 log_path=f"./{model_idx}/", eval_freq=logfreq,
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=logfreq,
        save_path=f"./{model_idx}/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    mlflow_callback = MFLow(freq=logfreq,
                            experiment_name=experiment_name,
                            log_path=f"./{model_idx}/")
    if not from_scratch:
        model_toload = find_latest(path=f"{model_idx}/")
        logger.info("loaded %s", model_toload)
        model = PPO.load(model_toload, env, device="cuda")
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=7e-4, device="cuda",
                    batch_size=256,
                    n_steps=4096
                    )

    model.learn(total_timesteps=9_000_000_000, callback=[eval_callback, checkpoint_callback, mlflow_callback])


if __name__ == "__main__":
    train_ttris()
