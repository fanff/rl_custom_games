import logging
import os
import sys

import mlflow
import numpy as np
import optuna
import click
from mlflow import ActiveRun
from optuna import Trial
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3 import dqn
from stable_baselines3 import ppo

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MLflowOutputFormat
from rl_custom_games.schedules import linear_schedule, pow_schedule
import torch as th

@click.command()
@click.option("--from_scratch", default=False, type=bool)
@click.option("--total_timestep", default=300_000, type=int, show_default=True)
@click.option("--board_height", default=8, type=int, show_default=True)
@click.option("--board_width", default=4, type=int, show_default=True)
@click.option("--brick_set", default="basic_ext", type=str, show_default=True)
@click.option("--max_step", default=2000, type=int, show_default=True)
def train_ttris(from_scratch, total_timestep, brick_set,
                board_height,
                board_width,
                max_step):
    logging.basicConfig(level=logging.INFO)

    # env = make_vec_env(CustomTetris, n_envs=n_envs)
    logger = logging.getLogger("ttrain")

    def objective(trial: Trial):
        rand = trial.suggest_categorical("rand", [42,314])
        n_envs = trial.suggest_categorical("n_envs", [8])
        # Categorical parameter
        learning_rate = 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

        batch_size = trial.suggest_categorical("batch_size", [32,64])
        n_steps: int = trial.suggest_categorical("n_steps", [1024])
        n_epochs = trial.suggest_categorical("n_epochs", [8,32])
        # trial.suggest_categorical("batch_size", [64])

        vf_net_size = trial.suggest_categorical("vf_net_size", [4,32])
        pi_net_size = trial.suggest_categorical("pi_net_size", [4,32])

        #
        mlflow_client = mlflow.MlflowClient()

        with ActiveRun(mlflow_client.create_run("0")) as active_run:
            run_id = active_run.info.run_id

            def log_param(k,v):
                mlflow_client.log_param(run_id,k,v)


            log_param("brick_set", brick_set)
            log_param("board_height", board_height)
            log_param("board_width", board_width)

            log_param("n_envs", n_envs)
            log_param("learning_rate", learning_rate)

            log_param("batch_size", batch_size)
            log_param("n_steps", n_steps)
            log_param("n_epochs", n_epochs)

            log_param("vf_net_size", vf_net_size)
            log_param("pi_net_size", pi_net_size)

            save_path = active_run.info.artifact_uri.replace("file:///E:/pychpj/testrl/", "")
            log_param("save_path", save_path)
            log_param("rand", rand)


            mlflow_output = MLflowOutputFormat(mlflow_client,run_id)

            loggers = Logger(
                folder=save_path,
                output_formats=[HumanOutputFormat(
                    os.path.join(save_path,"out.csv")),
                    mlflow_output,
                    TensorBoardOutputFormat(save_path)],
            )

            env = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)

            evalenv = VecTransposeImage(VecMonitor(
                VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
            ))

            class artifact_save(BaseCallback):
                def _on_step(self):
                    pass#

                    #mlflow.(os.path.join(save_path, "best_model.zip"),"best_model__.zip")



            eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                         n_eval_episodes=20,
                                         log_path=save_path, eval_freq=1000,
                                         deterministic=True, render=False,
                                         callback_on_new_best=artifact_save()
                                         )

            # checkpoint_callback = CheckpointCallback(
            #     save_freq=10_000,
            #     save_path=save_path,
            #     save_replay_buffer=True,
            #     save_vecnormalize=True,
            # )

            if not from_scratch:
                model_toload = find_latest(path=save_path)
                logger.info("loaded %s", model_toload)
                model = PPO.load(model_toload, env, device="cpu")
            else:



                policy_kwargs = dict(
                    features_extractor_kwargs={})
                policy_kwargs["net_arch"] = dict(pi=[pi_net_size], vf=[vf_net_size])
                policy_kwargs["normalize_images"] = False

                policy_kwargs["optimizer_class"] = RMSpropTFLike
                policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)


                model = PPO(ppo.MlpPolicy, env, device="cpu",
                           verbose=0,
                           learning_rate=pow_schedule(learning_rate,
                                                        learning_rate/2,
                                                        gamma=2),
                           ent_coef=0.01,
                           vf_coef=0.5,
                           clip_range=0.1,
                           batch_size=batch_size,
                           n_steps=n_steps,
                           n_epochs=n_epochs,
                           policy_kwargs=policy_kwargs,
                            #use_sde=True,
                           )
                print(model.policy.features_extractor)
                print(model.policy.mlp_extractor)

                # policy_kwargs = dict(features_extractor_kwargs={"features_dim": 128})
                # model = DQN(CnnPolicy, env, device="cuda",
                #             verbose=0,
                #             learning_rate=pow_schedule(learning_rate,
                #                                        learning_rate/10,
                #                                        gamma=2),
                #             buffer_size=1_000_000,  # 1e6
                #             learning_starts=50000,
                #             batch_size=batch_size,
                #             exploration_final_eps=0.1,
                #             exploration_fraction=0.01,
                #             tensorboard_log=save_path,
                #             tau=1.0,
                #             policy_kwargs=policy_kwargs
                #             )

                model.set_logger(loggers)

            log_param("model", model.__class__.__name__)
            log_param("model.policy", model.policy.__class__.__name__)
            log_param("model.policy.normalize_imgs", model.policy.normalize_images)


            model.learn(log_interval=8,
                        total_timesteps=total_timestep,
                        callback=[eval_callback, ],
                        )
        del model
        return 1

    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
    study.optimize(objective, n_trials=32,n_jobs=4)

if __name__ == "__main__":
    train_ttris()
