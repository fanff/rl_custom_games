import logging
import time

import click
from stable_baselines3 import PPO

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest


@click.command
@click.argument('model_path')
@click.option("--device", default="cuda", type=str,show_default=True)
@click.option("--max_step", default=100, type=int,show_default=True)
def eval_tetris(model_path,device,max_step):
    # logging.basicConfig(level=logging.DEBUG)
    brick_set = "traditional"
    board_height = 12
    board_width = 6

    current_model = ""

    while True:
        evalenv = CustomTetris(board_height, board_width, brick_set,max_step=max_step)

        modelpath = find_latest(path = model_path)
        if modelpath != current_model:
            model = PPO.load(modelpath, evalenv,device=device)
            current_model = model_path

        idx=0
        obs = evalenv.reset()
        while True:
            idx+=1
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = evalenv.step(action)
            print('\n'*3)
            print(modelpath,idx)
            evalenv.render("df")
            time.sleep(.04)

            # VecEnv resets automatically
            if done:
                break

if __name__ == "__main__":
    eval_tetris()