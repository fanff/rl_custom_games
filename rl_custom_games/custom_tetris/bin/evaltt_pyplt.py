import logging
import time
import mlflow
import click
from PIL.Image import Resampling
from stable_baselines3 import PPO, DQN
from PIL import Image
from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest

import matplotlib.pyplot as plt
print(plt.get_backend())
@click.command
@click.option("--device", default="cpu", type=str,show_default=True)
@click.option("--max_step", default=200, type=int,show_default=True)
def eval_tetris(device,max_step):
    # logging.basicConfig(level=logging.DEBUG)
    c = mlflow.MlflowClient()
    runs = mlflow.search_runs()
    run_id = runs.sort_values("start_time",ascending=False).iloc[0]["run_id"]

    print(run_id)
    r = c.get_run(run_id)
    print(r)
    params = r.data.to_dictionary()["params"]

    print(params)
    board_height =int(  params["board_height"] )
    board_width = int(  params["board_width"] )
    brick_set =  params["brick_set"]

    artifact_loc = params["save_path"].replace("file:///E:/pychpj/testrl/", "")
    print(artifact_loc)



    plt.ioff()
    plt.show(block=False)

    while True:
        evalenv = CustomTetris(board_height, board_width, brick_set,max_step=max_step)

        modelpath = find_latest(path = artifact_loc)
        print(modelpath, )
        if params["model"] == "PPO":
            model = PPO.load(modelpath, evalenv, device=device)
        elif params["model"] == "DQN":
            model = DQN.load(modelpath, evalenv, device=device)

        idx=0
        obs = evalenv.reset()
        img = Image.fromarray(obs.reshape((evalenv.output_width, evalenv.output_height))).resize((board_width , board_height ),Resampling.NEAREST)

        if idx == 0:
            img_obj = plt.imshow(img,cmap="gray")
            plt.axis('off')
            plt.axis("image")
        else:
            img_obj.set_data(img)

        while True:
            idx+=1
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = evalenv.step(action)

            img = Image.fromarray(obs.reshape((evalenv.output_width, evalenv.output_height))).resize(
                (board_width, board_height), Resampling.NEAREST)
            #plt.imshow(img)
            img_obj.set_data(img)
            #plt.draw()
            plt.pause(0.5)
            #print('\n'*3)
            #evalenv.render("df")
            #time.sleep(.01)

            # VecEnv resets automatically
            if done:
                break
        del model

if __name__ == "__main__":
    eval_tetris()