import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
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

