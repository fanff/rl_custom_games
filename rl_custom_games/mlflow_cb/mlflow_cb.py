import os
import time

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, KVWriter

class MFLow(BaseCallback):

    def __init__(
        self,
        log_path:str = "logs",
        experiment_name: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.parent = None
        self.name_prefix = experiment_name


        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path

        self.time_start_rollout = time.time()
        self.time_start_trainig = time.time()

        self.seen = []

    def _init_callback(self) -> None:
        pass


    def _on_step(self) -> bool:


        return True

    def _on_training_start(self) -> None:
        pass
    def _on_training_end(self) -> None:
        mlflow.log_metric("training_dur", time.time() - self.time_start_trainig, step=self.num_timesteps)
    def _on_rollout_start(self):
        self.time_start_rollout = time.time()
    def _on_rollout_end(self) -> None:
        mlflow.log_metric("roll_out_dur", time.time()-self.time_start_rollout, step=self.num_timesteps)



class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
            self,
            key_values,
            key_excluded,
            step=0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
                sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def log_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * (2 ** progress_remaining)

    return func
