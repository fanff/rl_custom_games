import os

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MFLow(BaseCallback):

    def __init__(
        self,
        log_path:str = "logs",
        freq:int = 10,
        experiment_name: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.freq = freq
        self.name_prefix = experiment_name
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path

        self.idx = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        import mlflow
        exp = mlflow.get_experiment_by_name(self.name_prefix)
        if exp is None:
            mlflow.create_experiment(self.name_prefix)
        exp = mlflow.get_experiment_by_name(self.name_prefix)
        self.current_run = mlflow.start_run(experiment_id=exp.experiment_id)

        # self.model
        # self.training_env
        # self.logger

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:

            npzfile = np.load(self.log_path+".npz")
            a = sorted(npzfile.files)
            avg_score = np.mean(npzfile.get("results")[self.idx])
            timesteps = npzfile.get("timesteps")[self.idx]
            avg_length = np.mean(npzfile.get("ep_lengths")[self.idx])

            mlflow.log_metric("avg_score",avg_score,step=timesteps)
            mlflow.log_metric("avg_length",avg_length,step=timesteps)

            mlflow.log_metric("learning_rate", self.model.learning_rate, step=self.num_timesteps)


            self.idx += 1

        return True
    #def _on_rollout_end(self) -> None:
    #    if self.n_calls % self.freq == 0:
    #        mlflow.log_metric("learning_rate", self.model.learning_rate, step=self.num_timesteps)