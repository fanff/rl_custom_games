import logging
import time
import mlflow
import click
from stable_baselines3 import PPO, DQN

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, find_latest


@click.command
@click.option("--device", default="cpu", type=str,show_default=True)
@click.option("--max_step", default=20000, type=int,show_default=True)
def eval_tetris(device,max_step):
    # logging.basicConfig(level=logging.DEBUG)
    c = mlflow.MlflowClient()
    runs = mlflow.search_runs()
    run_id = runs.sort_values("start_time",ascending=False).iloc[0]["run_id"]
    run_id = "edd3aa10b0be4010aa73dcfbbaff9d00"
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

    while True:
        evalenv = CustomTetris(board_height, board_width, brick_set,max_step=max_step)

        modelpath = find_latest(path = artifact_loc)
        #print(modelpath, )
        if params["model"] == "PPO":
            model = PPO.load(modelpath, evalenv, device=device)
        elif params["model"] == "DQN":
            model = DQN.load(modelpath, evalenv, device=device)

        idx=0
        total_reward = 0
        obs = evalenv.reset()
        while True:
            idx+=1
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = evalenv.step(action)
            #print('\n'*3)

            #evalenv.render("df")
            #time.sleep(.1)
            total_reward+=reward
            if done : print(modelpath,idx,total_reward) #;evalenv.render("df")
            # VecEnv resets automatically
            if done:
                break

if __name__ == "__main__":
    eval_tetris()