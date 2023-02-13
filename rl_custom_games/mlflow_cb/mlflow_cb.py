import os
import time

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
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

        self.time_start_rollout = time.time()
        self.time_start_trainig = time.time()

        self.seen = []

    def _init_callback(self) -> None:
        # Create folder if needed
        import mlflow
        exp = mlflow.get_experiment_by_name(self.name_prefix)
        if exp is None:
            mlflow.create_experiment(self.name_prefix)
        exp = mlflow.get_experiment_by_name(self.name_prefix)
        self.current_run = mlflow.start_run(experiment_id=exp.experiment_id)


    def _on_step(self) -> bool:
        #if len(self.logger.name_to_value) > 0 :
        #    print("l")
        if "train/n_updates" in self.logger.name_to_value:
            nup = self.logger.name_to_value["train/n_updates"]
            if nup not in self.seen:

                npzfile = np.load(self.log_path + ".npz")
                avg_score = np.mean(npzfile.get("results")[-1])
                avg_length = np.mean(npzfile.get("ep_lengths")[-1])
                mlflow.log_metric("avg_score", avg_score, step=self.num_timesteps)
                mlflow.log_metric("avg_length", avg_length, step=self.num_timesteps)


                for k,v in self.logger.name_to_value.items():
                    mlflow.log_metric(k, v, step=self.num_timesteps)

                self.seen.append(nup)

        if "eval/mean_ep_length" in self.logger.name_to_value:
            for k, v in self.logger.name_to_value.items():
                mlflow.log_metric(k, v, step=self.num_timesteps)
        if "eval/mean_reward" in self.logger.name_to_value:
            for k, v in self.logger.name_to_value.items():
                mlflow.log_metric(k, v, step=self.num_timesteps)


        return True

    def _on_training_start(self) -> None:
        self.time_start_trainig = time.time()
    def _on_training_end(self) -> None:
        mlflow.log_metric("training_dur", time.time() - self.time_start_trainig, step=self.num_timesteps)
    def _on_rollout_start(self):
        self.time_start_rollout = time.time()
    def _on_rollout_end(self) -> None:
        mlflow.log_metric("roll_out_dur", time.time()-self.time_start_rollout, step=self.num_timesteps)
        mlflow.log_metric("learning_rate", self.model.learning_rate, step=self.num_timesteps)

        #
        #


        for k,v in self.logger.name_to_count.items():
            mlflow.log_metric(k, v, step=self.num_timesteps)

        for k,v in self.logger.name_to_value.items():
            mlflow.log_metric(k, v, step=self.num_timesteps)

        if self.n_calls % self.freq == 0:
            mlflow.log_metric("learning_rate", self.model.learning_rate, step=self.num_timesteps)