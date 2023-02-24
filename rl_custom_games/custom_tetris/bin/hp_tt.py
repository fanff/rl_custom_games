import logging
import os
import sys
from typing import Callable, Tuple

import mlflow
import numpy as np
import optuna
import click
from gym.vector.utils import spaces
from mlflow import ActiveRun
from optuna import Trial
from optuna.study import StudyDirection
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import Logger, HumanOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3 import dqn
from stable_baselines3 import ppo

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MLflowOutputFormat, ActiveRunWrapper
from rl_custom_games.schedules import linear_schedule, pow_schedule
import torch as th
from torch import nn


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            features_dim,
            last_layer_dim_pi: int = 16,
            last_layer_dim_vf: int = 16,
    ):
        super().__init__()

        # padding  = 1 / 0

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,
                 convo_in_1=64,
                 convo_in_2=32):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # print(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), 1.0),
            nn.Conv2d(n_input_channels, convo_in_1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(convo_in_1, convo_in_2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            last_layer_dim_pi: int,
            last_layer_dim_vf,
            *args,
            **kwargs,
    ):
        self.last_layer_dim_pi = last_layer_dim_pi
        self.last_layer_dim_vf = last_layer_dim_vf
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.observation_space = observation_space

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim,
                                           last_layer_dim_pi=self.last_layer_dim_pi,
                                           last_layer_dim_vf=self.last_layer_dim_vf)


@click.command()
@click.option("--from_scratch", default=True, type=bool)
@click.option("--board_height", default=12, type=int, show_default=True)
@click.option("--board_width", default=5, type=int, show_default=True)
@click.option("--brick_set", default="basic_ext2", type=str, show_default=True)
@click.option("--max_step", default=500, type=int, show_default=True)
def train_ttris(from_scratch, brick_set,
                board_height,
                board_width,
                max_step):
    logging.basicConfig(level=logging.INFO)

    # env = make_vec_env(CustomTetris, n_envs=n_envs)
    logger = logging.getLogger("ttrain")

    def objective(trial: Trial):
        rand = trial.suggest_categorical("rand", [42, 314])
        n_envs = trial.suggest_categorical("n_envs", [8])
        # Categorical parameter
        learning_rate = trial.suggest_categorical("learning_rate", [0.001,
                                                                    0.0001])  # 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

        batch_size = trial.suggest_categorical("batch_size", [32])
        n_steps: int = trial.suggest_categorical("n_steps", [512])
        n_epochs = trial.suggest_categorical("n_epochs", [8])
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2])

        gamma = trial.suggest_categorical("gamma", [0.99])
        total_timestep = trial.suggest_categorical("total_timestep", [2_000_000, 5_000_000, 10_000_000])

        cnn_feature_size = trial.suggest_categorical("cnn_feature_size", [32, 64])
        vf_net_size = trial.suggest_categorical("vf_net_size", [32, 64])
        pi_net_size = trial.suggest_categorical("pi_net_size", [32, 64])

        convo_in_1 = trial.suggest_categorical("convo_in_1", [16, 32, 64, 128])
        convo_in_2 = trial.suggest_categorical("convo_in_2", [16, 32, 64, 128])

        #
        mlflow_client = mlflow.MlflowClient()

        with ActiveRunWrapper(mlflow_client.create_run("0"),mlflow_client) as active_run:
            run_id = active_run.info.run_id

            def log_param(k, v):
                mlflow_client.log_param(run_id, k, v)

            log_param("brick_set", brick_set)
            log_param("board_height", board_height)
            log_param("board_width", board_width)
            log_param("max_step", max_step)


            log_param("n_envs", n_envs)
            log_param("learning_rate", learning_rate)

            log_param("batch_size", batch_size)
            log_param("n_steps", n_steps)
            log_param("n_epochs", n_epochs)

            log_param("clip_range", clip_range)

            log_param("vf_net_size", vf_net_size)
            log_param("pi_net_size", pi_net_size)
            log_param("cnn_feature_size", cnn_feature_size)
            log_param("gamma", gamma)
            log_param("convo_in_1", convo_in_1)
            log_param("convo_in_2", convo_in_2)

            save_path = active_run.info.artifact_uri.replace("s3://minio/yourfolder/", "mlruns/")
            save_path = save_path.replace("s3://yourbucketname/yourfolder/", "mlruns/")
            save_path = save_path.replace("file://", "")
            save_path = save_path.replace("mlflow-artifacts:", "someruns")

            log_param("save_path", save_path)
            log_param("rand", rand)

            os.makedirs(os.path.join(save_path), exist_ok=True)
            mlflow_output = MLflowOutputFormat(mlflow_client, run_id)

            loggers = Logger(
                folder=save_path,
                output_formats=[HumanOutputFormat(
                    os.path.join(save_path, "out.csv")),
                    mlflow_output,
                    TensorBoardOutputFormat(save_path)],
            )

            env = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)

            listof_tetrisbuilder = [lambda: CustomTetris(board_height, board_width, brick_set, max_step, seed=_) for _
                                    in range(n_envs)]
            evalenv = VecTransposeImage(VecMonitor(VecTetris(listof_tetrisbuilder)))

            class artifact_save(BaseCallback):
                def _on_step(self):
                    pass  #

                    # mlflow.(os.path.join(save_path, "best_model.zip"),"best_model__.zip")

            es_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30,
                                                     min_evals=10,
                                                     verbose=1)
            eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                         n_eval_episodes=20,
                                         log_path=save_path, eval_freq=1000,
                                         deterministic=True, render=False,
                                         callback_after_eval=es_cb
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
                model = PPO.load(model_toload, env, device="cuda:1")
            else:

                policy_kwargs = {}
                # policy_kwargs["net_arch"] = dict(pi=pi_net_size, vf=vf_net_nsize)
                # policy_kwargs["normalize_images"] = False
                policy_kwargs = dict(
                    normalize_images=False,
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=cnn_feature_size,
                                                   convo_in_1=convo_in_1,
                                                   convo_in_2=convo_in_2),

                    last_layer_dim_pi=vf_net_size,
                    last_layer_dim_vf=pi_net_size
                )
                # policy_kwargs["optimizer_class"] = RMSpropTFLike
                # policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)

                model = PPO(CustomActorCriticPolicy, env, device="cpu",
                            verbose=0,
                            learning_rate=pow_schedule(learning_rate,
                                                       learning_rate / 100,
                                                       gamma=2),
                            ent_coef=0.01,
                            vf_coef=0.5,
                            gamma=gamma,
                            clip_range=clip_range,
                            batch_size=batch_size,
                            n_steps=n_steps,
                            n_epochs=n_epochs,
                            policy_kwargs=policy_kwargs,
                            # use_sde=True,
                            )

                mpol = model.policy

                fe_num_params = sum(param.numel() for param in mpol.features_extractor.parameters() if param.requires_grad)
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
            #mlflow_client.set_terminated(run_id, 'FINISHED')

        del model
        return last_mean_reward

    study = optuna.create_study(sampler=optuna.samplers.QMCSampler(),  # BruteForceSampler(),
                                direction=StudyDirection.MAXIMIZE)
    study.optimize(objective, n_trials=2048, n_jobs=1)


if __name__ == "__main__":
    train_ttris()
