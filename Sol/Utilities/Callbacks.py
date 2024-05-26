import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, TensorBoardOutputFormat
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
import wandb
from torch.utils.tensorboard import SummaryWriter


class SaveReplayBufferCallback(BaseCallback):
    """
    Custom Callback for saving the replay buffer of SAC (and possibly other off-policy methods).
    A single buffer is about 126MB.
    """

    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(SaveReplayBufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # model_path = os.path.join(self.save_path, f"model_{self.n_calls}.zip")
            replay_buffer_path = os.path.join(self.save_path, f"replay_buffer_{self.n_calls}.pkl")

            # self.model.save(model_path)
            self.model.save_replay_buffer(replay_buffer_path)

            if self.verbose > 0:
                print(f"Model and replay buffer saved at step {self.n_calls}")

        return True


class FoundTargetsCallback(BaseCallback):
    """
    Callback for plotting the number of found targets during training.
    """

    def __init__(self, log_dir, log_freq, verbose=1):
        super(FoundTargetsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self._log_freq = log_freq

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        if self.n_calls % self._log_freq == 0:
            found_targets = self.locals["infos"][0]["found_targets"]
            self.tb_formatter.writer.add_scalar("found_targets", found_targets)

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
        self.hparams = self.model.get_parameters()

        # Create a TensorBoard writer
        log_dir = self.model.tensorboard_log
        self.writer = SummaryWriter(log_dir)

        # Log hyperparameters
        for key, value in self.hparams.items():
            self.writer.add_text("hyperparameters", f"{key}: {value}")

        self.writer.flush()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        """
        This method is called when the training ends.
        It closes the TensorBoard writer.
        """
        if self.writer is not None:
            self.writer.close()


class SummaryWriterCallback2(BaseCallback):
    """
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    """

    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        """
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        """
        log_dir = self.model.tensorboard_log
        self.writer = SummaryWriter(log_dir)

        if self.n_calls % self._log_freq == 0:
            rewards = self.locals['my_custom_info_dict']['my_custom_reward']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i + 1),
                                                    rewards[i],
                                                    self.n_calls)

        # self.writer.add_scalar('reward', reward, steps_counter)

        return True
