import logging

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from rl_custom_games.custom_tetris.custom_tetris import CustomTetris
from rl_custom_games.mlflow_cb.mlflow_cb import MFLow


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # env = CustomTetris()
    # check_env(env, warn=True, skip_render_check=False)
    # quit()

    model_idx = "logs/ttppo2"
    env = make_vec_env(CustomTetris, n_envs=8)
    logfreq = 1000


    evalenv = CustomTetris()

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
                            log_path=f"./{model_idx}/")

    #model_toload = find_latest(path=f"{model_idx}/")
    #model = PPO.load(model_toload, env)
    #model.learning_rate=7e-4
    model = PPO("MlpPolicy", env, verbose=1,learning_rate=7e-4,)
    #model = A2C("MlpPolicy", env,
    #            learning_rate=7e-4,
    #            n_steps = 10,
    #            use_sde=False,
    #            verbose=1)

    model.learn(total_timesteps=300_000_000,callback=[eval_callback,checkpoint_callback,mlflow_callback])
    model.save("ttmodel")





    evalenv = CustomTetris()
    model = A2C.load("ttmodel",evalenv)
    obs = evalenv.reset()
    for i in range(100):
        print(i)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = evalenv.step(action)
        evalenv.render("df")
        # VecEnv resets automatically
        if done:
          obs = evalenv.reset()