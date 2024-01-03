import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
import wandb

class FoundTargetsCallback(BaseCallback):
    """
    Callback for plotting the number of found targets during training.

    """
    def __init__(self, log_dir, verbose=1):
        super(FoundTargetsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []

    def _on_step(self) -> bool:
        return True

    def _on_episode_end(self) -> None:
        print(self.model.ep_info_buffer)
        if self.model.ep_info_buffer:
            episode_info = self.model.ep_info_buffer[0]
            episode_rewards = episode_info.get('found_targets', None)
            print(episode_rewards)

            # Log episode rewards to TensorBoard
            if episode_rewards is not None:
                self.episode_rewards.append(episode_rewards[-1])
                self.logger.record('train/found_targets', episode_rewards[-1])
                wandb.log({'found_targets': episode_rewards[-1]})
                print("Found targets: ", episode_rewards[-1])


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "ent_coef": self.model.ent_coef,
            "clip_range": self.model.clip_range,
            "n_epochs": self.model.n_epochs,
            "n_steps": self.model.n_steps,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "gae_lambda": self.model.gae_lambda,
            "policy_kwargs": self.model.policy_kwargs,
            "policy": self.model.policy,
            "n_envs": self.model.n_envs,

        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
            "train/entropy_loss": 0.0,
            "train/policy_loss": 0.0,
            "train/approx_kl": 0.0,
            "train/clip_fraction": 0.0,
            "train/clip_range": 0.0,
            "train/n_updates_total": 0,
            "train/learning_rate": 0.0,
            "train/found_targets": 0.0,
            "train/ep_rew_mean": 0.0,
            "train/ep_rew_std": 0.0,
            "train/ep_len_mean": 0.0,
            "train/ep_len_std": 0.0,
            "train/success_rate": 0.0,
            "train/success_rate_std": 0.0,
            "train/success_rate_mean": 0.0,
            "train/episodes": 0.0,
            "train/time_elapsed": 0.0,
            "train/total_timesteps": 0.0,
            "train/total_updates": 0.0,
            "train/explained_variance": 0.0,
            "train/n_updates": 0.0,
            "train/serial_timesteps": 0.0,
            "train/serial_episodes": 0.0,
            "train/ep_rew_max": 0.0,
            "train/ep_rew_min": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True