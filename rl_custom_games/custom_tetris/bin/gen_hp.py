import logging
import os

import mlflow

import optuna
import click

from optuna import Trial
from optuna.study import StudyDirection
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

from rl_custom_games.custom_tetris.bin.hp_A2C_run import A2C_run
from rl_custom_games.custom_tetris.bin.hp_dqn_run import DQN_run
from rl_custom_games.mlflow_cb.mlflow_cb import MLflowOutputFormat, ActiveRunWrapper
from rl_custom_games.optuna_ext.utils import get_optuna_storage
import torch as th




@click.command()
@click.option("--board_height", default=20, type=int, show_default=True)
@click.option("--board_width", default=4, type=int, show_default=True)
@click.option("--brick_set", default="basic", type=str, show_default=True)
@click.option("--max_step", default=50, type=int, show_default=True)
@click.option("--device", default="cuda", type=str, show_default=True)
@click.option("--pidx", default=0, type=int, show_default=True)
def train_ttris(brick_set,
                board_height,
                board_width,
                max_step,
                device,
                pidx):
    logging.basicConfig(level=logging.INFO)

    if device == "cuda:":
        totaldevicecount = th.cuda.device_count()
        cuda_device_select = pidx % totaldevicecount
        device = f"cuda:{cuda_device_select}"
        print("getting into device ", device)

    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "0")

    def objective(trial: Trial):
        mlflow_client = mlflow.MlflowClient()

        with ActiveRunWrapper(mlflow_client.create_run(mlflow_experiment_id), mlflow_client) as active_run:
            run_id = active_run.info.run_id

            trial.set_user_attr("mlflow_run_id", run_id)
            trial.set_user_attr("mlflow_run_name", active_run.info.run_name)

            def log_param(k, v):
                mlflow_client.log_param(run_id, k, v)

            def pick_cat(name, opts):
                v = trial.suggest_categorical(name, opts)
                log_param(name, v)
                return v

            log_param("brick_set", brick_set)
            log_param("board_height", board_height)
            log_param("board_width", board_width)
            log_param("max_step", max_step)

            save_path = active_run.info.artifact_uri.replace("s3://minio/yourfolder/", "mlruns/")
            save_path = save_path.replace("s3://yourbucketname/yourfolder/", "mlruns/")
            save_path = save_path.replace("file:///", "")
            save_path = save_path.replace("mlflow-artifacts:", "someruns")

            log_param("save_path", save_path)

            os.makedirs(os.path.join(save_path), exist_ok=True)
            mlflow_output = MLflowOutputFormat(mlflow_client, run_id)

            loggers = Logger(
                folder=save_path,
                output_formats=[HumanOutputFormat(
                    os.path.join(save_path, "out.csv")),
                    mlflow_output,
                    TensorBoardOutputFormat(save_path)],
            )
            run = DQN_run
            last_mean_reward = run(pick_cat, log_param, loggers,save_path,
                                       board_height, board_width, brick_set, max_step, device
                                       )



        return last_mean_reward

    study = optuna.create_study(load_if_exists=True,
                                study_name="tetris_a2c_15",
                                sampler=optuna.samplers.QMCSampler(),  # BruteForceSampler(),
                                direction=StudyDirection.MAXIMIZE,
                                storage=get_optuna_storage())
    study.optimize(objective, n_trials=10, n_jobs=1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    train_ttris()
