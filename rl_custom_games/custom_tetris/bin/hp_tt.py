import logging
import sys

import mlflow
import numpy as np
import optuna
import click
from optuna import Trial
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.ppo import CnnPolicy

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MLflowOutputFormat, log_schedule


@click.command()
@click.option("--n_envs", default=1, type=int, show_default=True)
@click.option("--experiment_name", default="default")
@click.option("--parallel_env", default=False, type=bool)
@click.option("--logfreq", default=1000, type=int)
@click.option("--from_scratch", default=False, type=bool)
@click.option("--total_timestep", default=3_000_000, type=int, show_default=True)
@click.option("--board_height", default=10, type=int, show_default=True)
@click.option("--board_width", default=6, type=int, show_default=True)
@click.option("--brick_set", default="traditional", type=str, show_default=True)
@click.option("--max_step", default=2000, type=int, show_default=True)
def train_ttris(n_envs, experiment_name, parallel_env, logfreq, from_scratch, total_timestep, brick_set,
                board_height,
                board_width,
                max_step):
    logging.basicConfig(level=logging.INFO)

    # env = make_vec_env(CustomTetris, n_envs=n_envs)
    logger = logging.getLogger("ttrain")

    def objective(trial: Trial):

        # Categorical parameter
        learning_rate = 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

        batch_size = 64  # trial.suggest_categorical("batch_size", [64])

        with mlflow.start_run(experiment_id=617181189026246617) as active_run:

            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)

            save_path = active_run.info.artifact_uri.replace("mlflow-artifacts:", "mlruns")
            mlflow.log_param("save_path", save_path)

            mlflow_output = MLflowOutputFormat()
            loggers = Logger(
                folder=save_path,
                output_formats=[HumanOutputFormat("out.csv"), mlflow_output],
            )

            if parallel_env:
                env = SubprocVecEnv([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
            else:
                env = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)

            evalenv = VecTransposeImage(VecMonitor(
                VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
            ))

            eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                         n_eval_episodes=20,
                                         log_path=save_path, eval_freq=10000,
                                         deterministic=True, render=False
                                         )

            checkpoint_callback = CheckpointCallback(
                save_freq=10_000,
                save_path=save_path,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

            if not from_scratch:
                model_toload = find_latest(path=save_path)
                logger.info("loaded %s", model_toload)
                model = PPO.load(model_toload, env, device="cpu")
            else:

                # policy_kwargs = {}
                # policy_kwargs["optimizer_class"] = RMSpropTFLike
                # policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)
                # model = PPO("CnnPolicy", env, device="cpu",
                #            verbose=0,
                #            learning_rate=learning_rate,
                #            ent_coef=0.01,
                #            vf_coef=0.5,
                #            clip_range=0.1,
                #            batch_size=batch_size,
                #            n_steps=128,
                #            n_epochs=4,
                #            policy_kwargs=policy_kwargs
                #            )
                model = DQN("CnnPolicy", env, device="cuda",
                            verbose=0,
                            learning_rate=log_schedule(learning_rate),
                            buffer_size=1_000_000,  # 1e6
                            learning_starts=50000,
                            batch_size=batch_size,
                            exploration_final_eps=0.01,
                            )

                model.set_logger(loggers)

            model.learn(log_interval=8, total_timesteps=total_timestep, callback=[eval_callback, ])

        return 1

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=1)


VecTransposeImage
VecMonitor
CnnPolicy
if __name__ == "__main__":
    train_ttris()
