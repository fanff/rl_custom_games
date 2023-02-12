import time

import click
from stable_baselines3 import PPO

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest


@click.command
@click.argument('model_path')
def eval_tetris(model_path):
    idx=0
    while True:
        evalenv = CustomTetris()

        if idx%10 == 0:
            modelpath = find_latest(path = model_path)
            model = PPO.load(modelpath, evalenv)
        idx+=1
        obs = evalenv.reset()
        while True:

            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = evalenv.step(action)
            print('\n'*3)
            print(modelpath)
            evalenv.render("df")
            time.sleep(.04)

            # VecEnv resets automatically
            if done:
                break

if __name__ == "__main__":
    eval_tetris()