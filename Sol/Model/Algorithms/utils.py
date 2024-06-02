from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: initial learning rate
    :return: schedule that computes the current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def exponential_schedule(initial_value: float, decay_rate: float = 0.2) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.

    :param initial_value: initial learning rate
    :param decay_rate: decay rate (default: 0.2)
    :return: schedule that computes the current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * (decay_rate ** (1 - progress_remaining))

    return func


def lr_increase(initial_value: float, max_value: float, max_progress: float = 0.4) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate
    :param max_value: Maximum learning rate
    :param max_progress: The progress (as a fraction of total training) at which the learning rate reaches its maximum value
    :return: Schedule that computes the current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    if isinstance(max_value, str):
        max_value = float(max_value)
    if isinstance(max_progress, str):
        max_progress = float(max_progress)

    def func(progress_remaining: float) -> float:
        """
        Computes the current learning rate depending on remaining progress.

        :param progress_remaining: The remaining progress (0.0 to 1.0) of the training process
        :return: Current learning rate
        """
        if progress_remaining > 1.0 - max_progress:
            progress = (1.0 - progress_remaining) / max_progress
            return initial_value + (max_value - initial_value) * progress
        else:
            return max_value

    return func


def lrsched():
    def reallr(progress):
        lr = 0.003
        if progress < 0.85:
            lr = 0.0005
        if progress < 0.66:
            lr = 0.00025
        if progress < 0.33:
            lr = 0.0001
        return lr

    return reallr
