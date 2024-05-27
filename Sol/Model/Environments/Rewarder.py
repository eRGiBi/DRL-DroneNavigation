import numpy as np

"""
    Yet unused.
"""


def reaching_progress_reward(self):
    """
    Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning
    https://arxiv.org/pdf/2310.10943

    Approximation, as it is not explicitly stated.
    """

    reward = 0

    dist_to_cent = np.linalg.norm(self._current_position - self._target_points[self._current_target_index])

    if dist_to_cent <= self._threshold:
        self._current_target_index += 1
        reward += 3

    if self._current_target_index == len(self._target_points):
        self._is_done = True
        return 10

    dist_to_prev = np.linalg.norm(self._current_position - self._last_position)

    # Penalty term
    b = 0.01
    penalty_term = b * np.linalg.norm(self.pos[10:])

    # Collision penalty
    collision_penalty = -10.0 if self._has_collision_occurred() else 0.0

    reward += dist_to_prev - dist_to_cent - penalty_term + collision_penalty

    return reward


def calculate_progress_reward(self, pc_t, pc_t_minus_1, g1, g2):
    """
        Calculates the progress reward for the current and previous positions of the drone and the current and previous
        gate positions.
        Based on the reward function from https://arxiv.org/abs/2103.08624
        """

    def s(p):
        g_diff = g2 - g1
        return np.dot(p - g1, g_diff) / np.linalg.norm(g_diff) ** 2

    if pc_t_minus_1 is None:
        # Handle the edge case for the first gate
        rp_t = s(pc_t)
    else:
        rp_t = s(pc_t) - s(pc_t_minus_1)

    return rp_t

class BootstrappedImiVisionRewardCalculator:
    """
        https://arxiv.org/pdf/2403.12203
    """

    def __init__(self):
        self.lambda1 = 0.5
        self.lambda2 = 0.025
        self.lambda3 = 2e-4
        self.lambda4 = 5e-4
        self.c1 = 10
        self.c2 = 4

    def progress_reward(self, prev, now):
        return self.lambda1 * (prev - now)

    def perception_reward(self, delta_cam):
        return self.lambda2 * (self.lambda3 * (delta_cam ** 4))

    def command_smoothness_reward(self, a_t, a_t_minus_1):
        return -self.lambda3 * np.linalg.norm(a_t - a_t_minus_1)

    def body_rate_penalty(self, omega_t):
        return -self.lambda4 * np.linalg.norm(omega_t)

    def gate_passing_reward(self, passed):
        return self.c1 if passed else 0

    def collision_penalty(self, crashed):
        return -self.c2 if crashed else 0

    def calculate_reward(self, prev_dis, dis, delta_cam, a_t, a_t_minus_1, omega_t, passed, crashed):
        r_prog_t = self.progress_reward(prev_dis, dis)
        r_perc_t = self.perception_reward(delta_cam)
        r_act_t = self.command_smoothness_reward(a_t, a_t_minus_1)
        r_br_t = self.body_rate_penalty(omega_t)
        r_pass_t = self.gate_passing_reward(passed)
        r_crash_t = self.collision_penalty(crashed)

        total_reward = r_prog_t + r_perc_t + r_act_t + r_br_t + r_pass_t + r_crash_t
        return total_reward


class ChampRewardCalculator:
    """
        https://www.nature.com/articles/s41586-023-06419-4.pdf

    """

    def __init__(self):
        lambda1 = 1.0
        lambda2 = 0.02
        lambda3 = -10.0
        lambda4 = -2e-4
        lambda5 = -1e-4
        c1 = 5.0
        c2 = 0

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.c1 = c1
        self.c2 = c2

    def progress_reward(self, dGate_t_minus_1, dGate_t):
        return self.lambda1 * (dGate_t_minus_1 - dGate_t)

    def perception_reward(self, delta_cam):
        """delta_cam:
        This represents the angle between the camera's optical axis
        and the direction towards the center of the next gate.
        """
        return self.lambda2 * np.exp(self.lambda3 * (delta_cam ** 4))

    def command_smoothness_reward(self, a_t, a_t_minus_1, omega_t):
        return self.lambda4 * np.linalg.norm(omega_t) ** 2 + self.lambda5 * np.linalg.norm(a_t - a_t_minus_1) ** 2

    def collision_penalty(self, p_z, in_collision):
        return self.c1 if p_z < 0 or in_collision else 0

    def calculate_reward(self, dGate_t_minus_1, dGate_t, delta_cam, a_t, a_t_minus_1, omega_t, p_z, in_collision):
        r_prog_t = self.progress_reward(dGate_t_minus_1, dGate_t)
        r_perc_t = self.perception_reward(delta_cam)
        r_cmd_t = self.command_smoothness_reward(a_t, a_t_minus_1, omega_t)
        r_crash_t = self.collision_penalty(p_z, in_collision)

        total_reward = r_prog_t + r_perc_t + r_cmd_t - r_crash_t
        return total_reward
