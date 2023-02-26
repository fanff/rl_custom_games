from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, \
    StopTrainingOnNoModelImprovement

from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecMonitor

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import VecTetris, GroupedActionSpace

from rl_custom_games.schedules import pow_schedule
import torch as th
from torch import nn



def DQN_run(pick_cat, log_param, loggers, save_path,
            board_height, board_width, brick_set, max_step, device
            ):
    rand = pick_cat("rand", [42, 314])

    use_rmsprop = pick_cat("use_rmsprop", [True, False])

    format_as_onechannel = pick_cat("format_as_onechannel", [True])
    n_envs = pick_cat("n_envs", [8, 16, 24])
    learning_rate = pick_cat("learning_rate",
                             [0.01,
                              0.009])  # 0.001  # trial.suggest_float("learning_rate", 0.0001, 0.001)

    buffer_size = pick_cat("buffer_size", [500_000, 1_000_000])

    batch_size = pick_cat("batch_size", [32])
    total_timestep = pick_cat("total_timestep", [2_000_000, 5_000_000, 10_000_000])

    layer_1 = pick_cat("layer_1", [32,64])
    layer_2 = pick_cat("layer_2", [32,64])
    layer_3 = pick_cat("layer_3", [16,32])

    target_update_interval = pick_cat("target_update_interval", [500,1000, 2000, 3000])

    learning_starts = pick_cat("learning_starts", [5000])


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
        net_arch=[layer_1, layer_2,layer_3],

    )

    if use_rmsprop:
        policy_kwargs["optimizer_class"] = RMSpropTFLike
        policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)

    model = DQN(MlpPolicy, env, device=device,
                verbose=0,
                learning_rate=pow_schedule(learning_rate,
                                           learning_rate / 100,
                                           gamma=2),
                buffer_size=buffer_size,  # 1e6
                learning_starts=learning_starts,
                batch_size=batch_size,
                exploration_final_eps=0.02,
                exploration_fraction=0.01,
                target_update_interval=target_update_interval,
                tensorboard_log=save_path,
                tau=1.0,
                policy_kwargs=policy_kwargs
                )

    mpol = model.policy



    log_param("q_net_params", sum(param.numel() for param in mpol.q_net.parameters() if param.requires_grad))
    log_param("q_net_target_params", sum(param.numel() for param in mpol.q_net_target.parameters() if param.requires_grad))

    model.set_logger(loggers)

    log_param("model", model.__class__.__name__)
    log_param("model.policy", model.policy.__class__.__name__)

    model.learn(log_interval=8,
                total_timesteps=total_timestep,
                callback=[eval_callback, ],
                )

    last_mean_reward = eval_callback.last_mean_reward
    log_param("final_reward", last_mean_reward)

    del model
    return last_mean_reward
