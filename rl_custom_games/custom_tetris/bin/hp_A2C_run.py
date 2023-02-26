import os

import json

import gym

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import Logger, HumanOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import VecTetris, GroupedActionSpace

from rl_custom_games.schedules import pow_schedule
import torch as th
from torch import nn


class CustomFExtractor(BaseFeaturesExtractor):
    """

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, size=32) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(get_flattened_obs_dim(observation_space), size), nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)


def A2C_run(pick_cat, log_param, loggers, save_path,
            board_height, board_width, brick_set, max_step, device
            ):
    rand = pick_cat("rand", [42, 314])

    use_rmsprop = pick_cat("use_rmsprop", [True, False])

    activation_fct = pick_cat("activation", ["relu", "tanh"])
    activation_fct_dict = {"tanh": nn.Tanh, "relu": nn.ReLU}
    activation_fct = activation_fct_dict[activation_fct]

    format_as_onechannel = pick_cat("format_as_onechannel", [True])
    n_envs = pick_cat("n_envs", [8, 16, 24])
    learning_rate = pick_cat("learning_rate",
                             [0.0003,
                              0.0001])  # 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

    n_steps = pick_cat("n_steps", [5, 8])

    gamma = pick_cat("gamma", [0.99])
    total_timestep = pick_cat("total_timestep", [1_000_000])

    shared_layer_size = pick_cat("shared_layer_size", [64, 128])
    vf_size = pick_cat("vf_size", ["[64]", "[64,64]"])

    vf_size = json.loads(vf_size)
    pi_size = pick_cat("pi_size", ["[64]", "[64,64]"])
    pi_size = json.loads(pi_size)

    env = VecTetris([lambda: GroupedActionSpace(board_height, board_width, brick_set,
                                                max_step,
                                                format_as_onechannel=format_as_onechannel)] * n_envs)

    listof_tetrisbuilder = [lambda: GroupedActionSpace(board_height, board_width, brick_set, max_step, seed=_,
                                                       format_as_onechannel=format_as_onechannel) for _
                            in range(n_envs)]
    evalenv = (VecMonitor(VecTetris(listof_tetrisbuilder)))

    es_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=150,
                                             min_evals=10,
                                             verbose=1)

    eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                 n_eval_episodes=40,
                                 log_path=save_path, eval_freq=300,
                                 deterministic=True, render=False,
                                 callback_after_eval=es_cb
                                 )

    policy_kwargs = dict(
        normalize_images=False,
        net_arch=[shared_layer_size, dict(vf=vf_size,
                                          pi=pi_size)],
        activation_fn=activation_fct,

    )

    if use_rmsprop:
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
                use_rms_prop=use_rmsprop
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

    del model
    return last_mean_reward
