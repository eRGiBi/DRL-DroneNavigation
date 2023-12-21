import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    """
    def __init__(self, log_dir, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self._plot = None

    def _on_step(self) -> bool:
        # Log found targets information during training steps
        info_dict = self.model.ep_info_buffer[0]
        found_targets = info_dict.get('found_targets', None)
        if found_targets is not None:
            # Log the found_targets to TensorBoard
            self.logger.record('train/found_targets', found_targets)

        return True
