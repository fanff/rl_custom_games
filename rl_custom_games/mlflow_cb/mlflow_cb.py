import os
import time

import mlflow
import numpy as np
from mlflow.entities import Run, RunStatus
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

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


class ActiveRunWrapper(Run):  # pylint: disable=W0223
    """Wrapper around :py:class:`mlflow.entities.Run` to enable using Python ``with`` syntax."""

    def __init__(self, run,mlflow_client):
        Run.__init__(self, run.info, run.data)

        self.mlflow_client = mlflow_client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED

        self.mlflow_client.set_terminated(self.info.run_id, RunStatus.to_string(status))
        return exc_type is None
class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """
    def __init__(self,client,run_id):
        self.client = client
        self.run_id = run_id
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
                    self.client.log_metric(self.run_id,key, value, step=step)

