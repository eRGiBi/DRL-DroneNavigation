import numpy as np


class Rewarder():
    pass



class RewardCalculator:
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
