import logging
import os
import random
import sys
from typing import Callable, Tuple

import gym
import mlflow
import numpy as np
import optuna
import click
from dotenv import load_dotenv
from gym.vector.utils import spaces
from mlflow import ActiveRun
from optuna import Trial
from optuna.study import StudyDirection
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import Logger, HumanOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris, \
    GroupedActionSpace
from rl_custom_games.mlflow_cb.mlflow_cb import MLflowOutputFormat, ActiveRunWrapper
from rl_custom_games.optuna_ext.utils import get_optuna_storage
from rl_custom_games.schedules import linear_schedule, pow_schedule
import torch as th
from torch import nn


class CustomFExtractor(BaseFeaturesExtractor):
    """

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space,size=32) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

        self.linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(get_flattened_obs_dim(observation_space), size), nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)

@click.command()
@click.option("--from_scratch", default=True, type=bool)
@click.option("--board_height", default=14, type=int, show_default=True)
@click.option("--board_width", default=6, type=int, show_default=True)
@click.option("--brick_set", default="traditional", type=str, show_default=True)
@click.option("--max_step", default=50, type=int, show_default=True)
@click.option("--device", default="cuda", type=str, show_default=True)
@click.option("--pidx", default=0, type=int, show_default=True)
def train_ttris(from_scratch, brick_set,
                board_height,
                board_width,
                max_step,
                device,
                pidx):
    logging.basicConfig(level=logging.INFO)

    if device == "cuda:":
        totaldevicecount = th.cuda.device_count()
        cuda_device_select = pidx % totaldevicecount
        device = f"cuda:{cuda_device_select}"
        print("getting into device ", device)

    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "0")
    def objective(trial: Trial):

        mlflow_client = mlflow.MlflowClient()

        with ActiveRunWrapper(mlflow_client.create_run(mlflow_experiment_id), mlflow_client) as active_run:
            run_id = active_run.info.run_id

            trial.set_user_attr("mlflow_run_id", run_id)
            trial.set_user_attr("mlflow_run_name", active_run.info.run_name)

            def log_param(k, v):
                mlflow_client.log_param(run_id, k, v)

            def pick_cat(name, opts):
                v = trial.suggest_categorical(name, opts)
                log_param(name, v)
                return v

            log_param("brick_set", brick_set)
            log_param("board_height", board_height)
            log_param("board_width", board_width)
            log_param("max_step", max_step)



            rand = pick_cat("rand", [42, 314])
            format_as_onechannel = pick_cat("format_as_onechannel", [True])
            n_envs = pick_cat("n_envs", [8, 16])
            learning_rate = pick_cat("learning_rate",
                                                      [0.001,
                                                       0.0001])  # 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

            n_steps = pick_cat("n_steps", [5, 8])

            gamma = pick_cat("gamma", [0.99])
            total_timestep = pick_cat("total_timestep", [2_000_000, 5_000_000, 10_000_000])

            shared_layer_size = pick_cat("shared_layer_size", [64, 128])
            vf_size = pick_cat("vf_size", [32, 64])
            pi_size = pick_cat("pi_size", [32, 64])

            save_path = active_run.info.artifact_uri.replace("s3://minio/yourfolder/", "mlruns/")
            save_path = save_path.replace("s3://yourbucketname/yourfolder/", "mlruns/")
            save_path = save_path.replace("file:///", "")
            save_path = save_path.replace("mlflow-artifacts:", "someruns")

            log_param("save_path", save_path)

            os.makedirs(os.path.join(save_path), exist_ok=True)
            mlflow_output = MLflowOutputFormat(mlflow_client, run_id)

            loggers = Logger(
                folder=save_path,
                output_formats=[HumanOutputFormat(
                    os.path.join(save_path, "out.csv")),
                    mlflow_output,
                    TensorBoardOutputFormat(save_path)],
            )

            env = VecTetris([lambda: GroupedActionSpace(board_height, board_width, brick_set,
                                                  max_step,
                                                  format_as_onechannel=format_as_onechannel)] * n_envs)

            listof_tetrisbuilder = [lambda: GroupedActionSpace(board_height, board_width, brick_set, max_step, seed=_,
                                                         format_as_onechannel=format_as_onechannel) for _
                                    in range(n_envs)]
            evalenv = VecTransposeImage(VecMonitor(VecTetris(listof_tetrisbuilder)))

            es_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50,
                                                     min_evals=10,
                                                     verbose=1)

            eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                         n_eval_episodes=20,
                                         log_path=save_path, eval_freq=1000,
                                         deterministic=True, render=False,
                                         callback_after_eval=es_cb
                                         )

            # policy_kwargs["net_arch"] = dict(pi=pi_net_size, vf=vf_net_nsize)
            # policy_kwargs["normalize_images"] = False
            # policy_kwargs = dict(
            #    normalize_images=False,
            #    features_extractor_class=CustomCNN,
            #    features_extractor_kwargs=dict(features_dim=cnn_feature_size,
            #                                   convo_in_1=convo_in_1,
            #                                   convo_in_2=convo_in_2),
            #
            #    last_layer_dim_pi=vf_net_size,
            #    last_layer_dim_vf=pi_net_size
            # )
            activation_fn = nn.ReLU
            policy_kwargs = dict(
                normalize_images=False,
                net_arch=[shared_layer_size, dict(vf=[vf_size],
                                                  pi=[pi_size])],
                activation_fn = activation_fn,
                #features_extractor_class=CustomFExtractor,
                #features_extractor_kwargs=dict(size=64),

                # last_layer_dim_pi=vf_net_size,
                # last_layer_dim_vf=pi_net_size
            )
            policy_kwargs["optimizer_class"] = RMSpropTFLike
            policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)


            model = A2C(MlpPolicy, env, device=device,
                        verbose=0,
                        learning_rate=pow_schedule(learning_rate,
                                                   learning_rate / 100,
                                                   gamma=2),
                        ent_coef=0.01,
                        vf_coef=0.5,
                        gamma=gamma,
                        n_steps=n_steps,
                        policy_kwargs=policy_kwargs,
                        # use_sde=True,
                        )

            mpol = model.policy

            fe_num_params = sum(
                param.numel() for param in mpol.features_extractor.parameters() if param.requires_grad)
            mlp_num_params = sum(param.numel() for param in mpol.mlp_extractor.parameters() if param.requires_grad)
            vn_num_params = sum(param.numel() for param in mpol.value_net.parameters() if param.requires_grad)
            fn_num_params = sum(param.numel() for param in mpol.action_net.parameters() if param.requires_grad)

            log_param("fe_num_params", fe_num_params)
            log_param("mlp_num_params", mlp_num_params)
            log_param("vn_num_params", vn_num_params)
            log_param("fn_num_params", fn_num_params)

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

            last_mean_reward = eval_callback.last_mean_reward
            log_param("final_reward", last_mean_reward)
            # mlflow_client.set_terminated(run_id, 'FINISHED')

        del model
        return last_mean_reward

    study = optuna.create_study(load_if_exists=True,
                                study_name="tetris_a2c_9",
                                sampler=optuna.samplers.QMCSampler(),  # BruteForceSampler(),
                                direction=StudyDirection.MAXIMIZE,
                                storage=get_optuna_storage())
    study.optimize(objective, n_trials=10, n_jobs=1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    train_ttris()
