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

    use_rmsprop = pick_cat("use_rmsprop", [False])

    format_as_onechannel = pick_cat("format_as_onechannel", [True])
    n_envs = pick_cat("n_envs", [4])
    learning_rate = pick_cat("learning_rate",
                             [0.0001,])

    learning_rate_div = pick_cat("learning_rate_div",
                             [10])

    buffer_size = pick_cat("buffer_size", [10_000_000])

    batch_size = pick_cat("batch_size", [128,64])
    total_timestep = pick_cat("total_timestep", [20_000_000])

    layer_1 = pick_cat("layer_1", [512,])
    layer_2 = pick_cat("layer_2", [256,])
    layer_3 = pick_cat("layer_3", [256,])
    layer_4 = pick_cat("layer_4", [256,])

    target_update_interval = pick_cat("target_update_interval", [100000])

    learning_starts = pick_cat("learning_starts", [5000])
    
    gamma = pick_cat("gamma", [0.99])
    fail_limit = 10
    env = VecTetris([lambda: GroupedActionSpace(board_height, board_width, brick_set,
                                                max_step,
                                                format_as_onechannel=format_as_onechannel,
                                                fail_limit=fail_limit)] * n_envs)

    listof_tetrisbuilder = [lambda: GroupedActionSpace(board_height, board_width, brick_set, max_step, seed=_,
                                                       format_as_onechannel=format_as_onechannel,
                                                       fail_limit=fail_limit) for _
                            in range(n_envs)]
    evalenv = (VecMonitor(VecTetris(listof_tetrisbuilder)))

    es_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20000,
                                             min_evals=50,
                                             verbose=1)

    eval_callback = EvalCallback(evalenv, best_model_save_path=save_path,
                                 n_eval_episodes=1,
                                 log_path=save_path, eval_freq=1000,
                                 deterministic=True, render=False,
                                 callback_after_eval=es_cb
                                 )

    policy_kwargs = dict(
        normalize_images=False,
        net_arch=[layer_1,layer_2,24],

    )

    if use_rmsprop:
        policy_kwargs["optimizer_class"] = RMSpropTFLike
        policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)

    model = DQN(MlpPolicy, env, device=device,
                verbose=2,
                learning_rate=pow_schedule(learning_rate,
                                           learning_rate / learning_rate_div,
                                           gamma=2),
                buffer_size=buffer_size,  # 1e6
                learning_starts=learning_starts,
                batch_size=batch_size,
                exploration_final_eps=0.05,
                exploration_fraction=0.1,
                target_update_interval=target_update_interval,
                tensorboard_log=save_path,
                tau=1.0,
                gamma=gamma,
                policy_kwargs=policy_kwargs
                )

    mpol = model.policy



    log_param("q_net_params", sum(param.numel() for param in mpol.q_net.parameters() if param.requires_grad))
    log_param("q_net_target_params", sum(param.numel() for param in mpol.q_net_target.parameters() if param.requires_grad))

    model.set_logger(loggers)

    log_param("model", model.__class__.__name__)
    log_param("model.policy", model.policy.__class__.__name__)

    model.learn(log_interval=24,
                total_timesteps=total_timestep,
                callback=[eval_callback, ],

                )

    last_mean_reward = eval_callback.last_mean_reward
    log_param("final_reward", last_mean_reward)

    del model
    return last_mean_reward
