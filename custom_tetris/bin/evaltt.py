import os
import time

from stable_baselines3 import A2C, PPO

import logging

from custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest

if __name__ == "__main__":
    for i in range(120):
        print()

    evalenv = CustomTetris()
    idx=0
    while True:
        evalenv = CustomTetris()

        if idx%10 == 0:
            modelpath = find_latest(path ="../../logs/ttppo35/")
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

