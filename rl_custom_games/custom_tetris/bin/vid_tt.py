

import imageio
import logging
import time

import click
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest


@click.command
@click.argument('model_path')
@click.option("--device", default="cpu", type=str,show_default=True)
@click.option("--max_step", default=100, type=int,show_default=True)
def eval_tetris(model_path,device,max_step):
    # logging.basicConfig(level=logging.DEBUG)
    brick_set = "traditional"
    board_height = 14
    board_width = 6
    video_folder = "logs/videos/"
    current_model = ""
    evalenv = CustomTetris(board_height, board_width, brick_set, max_step=max_step)




    # Record the video starting at the first step


    model = DQN.load(model_path, evalenv, device=device)
    obs = model.env.reset()

    for _ in range(max_step + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()

if __name__ == "__main__":
    eval_tetris()









