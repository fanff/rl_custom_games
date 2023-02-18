import logging
import optuna
import click
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.ppo import CnnPolicy

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest, VecTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MFLow


@click.command()
@click.option("--model_idx", default="logs/default")
@click.option("--n_envs", default=4, type=int, show_default=True)
@click.option("--experiment_name", default="default")
@click.option("--parallel_env", default=False, type=bool)
@click.option("--logfreq", default=1000, type=int)
@click.option("--from_scratch", default=False, type=bool)
@click.option("--total_timestep", default=300_000_000, type=int, show_default=True)
@click.option("--board_height", default=16, type=int, show_default=True)
@click.option("--board_width", default=8, type=int, show_default=True)
@click.option("--brick_set", default="traditional", type=str, show_default=True)
@click.option("--max_step", default=2000, type=int, show_default=True)
def train_ttris(model_idx, n_envs, experiment_name, parallel_env, logfreq, from_scratch,total_timestep, brick_set,
                board_height,
                board_width,
                max_step):
    logging.basicConfig(level=logging.INFO)

    # env = make_vec_env(CustomTetris, n_envs=n_envs)
    logger = logging.getLogger("ttrain")

    logger.info("model_idx: %s, n_envs: %s, experiment_name: %s, parallel_env: %s, logfreq: %s, from_scratch: %s",
                model_idx, n_envs, experiment_name, parallel_env, logfreq, from_scratch)
    if parallel_env:
        env = SubprocVecEnv([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
    else:
        env = VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)

    evalenv = VecTransposeImage(VecMonitor(
        VecTetris([lambda: CustomTetris(board_height, board_width, brick_set, max_step)] * n_envs)
    ))

    mlflow_callback = MFLow(experiment_name=experiment_name,
                            log_path=f"./{model_idx}/")


    eval_callback = EvalCallback(evalenv, best_model_save_path=f"./{model_idx}/",
                                 n_eval_episodes=20,
                                 log_path=f"./{model_idx}/", eval_freq=8192/n_envs,
                                 deterministic=True, render=False,
                                 callback_after_eval=mlflow_callback)
    mlflow_callback.parent = eval_callback


    checkpoint_callback = CheckpointCallback(
        save_freq=logfreq,
        save_path=f"./{model_idx}/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    if not from_scratch:
        model_toload = find_latest(path=f"{model_idx}/")
        logger.info("loaded %s", model_toload)
        model = PPO.load(model_toload, env, device="cuda")
    else:
        #policy_kwargs={}
        #policy_kwargs["optimizer_class"] = RMSpropTFLike
        #policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)
        #model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, device="cuda",
        #            batch_size=128,
        #            n_steps=4096*4,
        #            policy_kwargs=policy_kwargs
        #            )



        policy_kwargs = {}
        policy_kwargs["optimizer_class"] = RMSpropTFLike
        policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)
        model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-2, device="cuda",
                    ent_coef=0.01,
                    vf_coef=0.5,
                    clip_range=0.1,
                    batch_size=256,
                    n_steps=128,
                    n_epochs=4,
                    policy_kwargs=policy_kwargs
                    )

    model.learn(total_timesteps=300_000, callback=[eval_callback, ])

VecTransposeImage
VecMonitor
CnnPolicy
if __name__ == "__main__":
    train_ttris()

