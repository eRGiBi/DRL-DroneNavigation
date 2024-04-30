import csv
import gzip
import os
import math
import copy

import inspect
import threading
from datetime import datetime

import gym
import pandas as pd
import torch
from gymnasium import spaces
import numpy as np
import pybullet as p
from stable_baselines3.common.running_mean_std import RunningMeanStd

from Sol.PyBullet.enums import DroneModel, Physics, ActionType, ObservationType
from Sol.PyBullet.GymPybulletDronesMain import *
from Sol.PyBullet.BaseSingleAgentAviary import BaseSingleAgentAviary
from Sol.PyBullet.FlyThruGateAviary import FlyThruGateAviary
from gymnasium.spaces.space import Space
from Sol.Utilities.position_generator import PositionGenerator


class PBDroneEnv(
    # BaseAviary,
    # FlyThruGateAviary,
    BaseSingleAgentAviary,
):

    def __init__(self,
                 target_points, threshold, discount, max_steps, aviary_dim,
                 save_model=False, save_folder=None,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 vision_attributes=False,
                 user_debug_gui=False,
                 obstacles=False,
                 random_spawn=False
                 ):

        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self.OBS_TYPE = obs
        self._OBS_TYPE = obs

        self._target_points = np.array(target_points)
        # self._reached_targets = np.zeros(len(self._target_points), dtype=bool)
        self._threshold = threshold
        self._discount = discount
        self._max_steps = max_steps
        self._aviary_dim = aviary_dim
        self._x_low, self._y_low, self._z_low, self._x_high, self._y_high, self._z_high = aviary_dim
        self.random_spawn = random_spawn
        print("AVIARY DIM", self._aviary_dim)

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         # vision_attributes=vision_attributes,
                         )

        self._current_position = self.INIT_XYZS[0]
        self._last_position = None

        self._steps = 0
        self.total_steps = 0
        # self.steps_since_last_target = 0

        self._last_action = np.zeros(4, dtype=np.float32)

        # Smallest possible value for the data type to avoid underflow exceptions
        # self.eps = np.finfo(self._last_action.dtype).eps
        self.eps = np.finfo(self._last_action.dtype).tiny

        self._distance_to_target = np.linalg.norm(self._current_position - target_points[0])
        self._prev_distance_to_target = np.linalg.norm(self._current_position - target_points[0])
        self._current_target_index = 0
        self.just_found = False

        self._is_done = False

        self.CLIENT = self.getPyBulletClient()
        self.target_visual = []

        self.save_folder = save_folder
        self.file_path = os.path.join('Sol/rollouts/', 'rollout_' + str(sum(len(files) for _, _, files in os.walk('Sol/rollouts/'))) + '.txt')
        self.lock = threading.Lock()

        if save_folder is not None:
            self.save_model(save_folder)

        if gui:
            self.show_targets()

        # self._addObstacles()

        if self.random_spawn:
            self.init_position_generator = PositionGenerator(self._aviary_dim, 0.5)

    def step(self, action):
        """Applies the given action to the environment."""

        obs, reward, terminated, truncated, info = (
            super().step(action)
        )
        # print("ACTION", action)
        # print("OBS", obs)
        # print("pos", self.pos[0])
        # print("REWARD", reward)
        # print("TERMINATED", terminated)

        #
        # if True and len(obs) > 0:
        #     with open(self.file_path, mode='a+') as f:
        #         with self.lock: # Doesnt work even with thread locking with multiple envs
        #             for x in obs.tolist():
        #                 f.write(str(np.format_float_positional(np.float32(x), unique=False, precision=32)) + ",")
        #             f.write(str(reward))
        #             f.write("\n")
        #     f.close()

        # self.total_steps += 1

        if not terminated:
            self._steps += 1
            self._last_action = action
            self._last_position = copy.deepcopy(self._current_position)
            self._current_position = self.pos[0] #+ self.eps

            # Calculate the Euclidean distance between the drone and the next target
            self._distance_to_target = abs(np.linalg.norm(
                self._target_points[self._current_target_index] - self._current_position))

            # distance_to_target = self.distance_between_points(self._computeObs()[:3],
            #                                                   self._target_points[self._current_target_index])

        if self.GUI and self._distance_to_target <= self._threshold:
            self.remove_target()

        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        """Returns the action space of the environment."""

        return spaces.Box(low=-1 * np.ones(4, dtype=np.float32),
                          high=np.ones(4, dtype=np.float32),
                          shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""

        # return spaces.Box(low=np.array([self._x_low, self._y_low, 0,
        #                                 -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        #                   high=np.array([self._x_high, self._y_high, self._z_high,
        #                                  1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        #                   dtype=np.float32
        #                     )

        return spaces.Box(low=np.array([1, 1, 0,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
                          high=np.array([1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                          dtype=np.float32
                          )

    def _computeObs(self):
        """
        Returns the current observation of the environment.

        Kinematic observation of size 12.

        """
        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        # obs = self._getDroneStateVector(0)

        obs = obs #+ self.eps
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )

        # print("x", obs[0], "y", obs[1], "z", obs[2])
        # print("roll", obs[7], "pitch", obs[8], "yaw", obs[9])
        # print("vel_x", obs[10], "vel_y", obs[11], "vel_z", obs[12])
        # print("ang_vel_x", obs[13], "ang_vel_y", obs[14], "ang_vel_z", obs[15])

        # #TODO: NORMALIZation check
        # ret = np.array([ret[0]/self._x_high, ret[1]/self._y_high, ret[2]/self._z_high,
        #                 ret[3], ret[4], ret[5], ret[6], ret[7], ret[8], ret[9], ret[10], ret[11]])

        try:
            return ret #.astype('float32')
        except FloatingPointError as e:
            print("Error in _computeObs():", ret)
            print(f"Underflow error: {e}")
            # return np.zeros_like(ret).astype('float32')
            return np.clip(ret, np.finfo(np.float32).min, np.finfo(np.float32).max)#.astype('float32')

    def _clipAndNormalizeState(self, state):
        """
        Normalizes a drone's state to the [-1,1] range.

        np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        """

        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1
        self.EPISODE_LEN_SEC = 1
        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        # clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        # clipped_pos_x = np.clip(state[0], self._aviary_dim[0], self._aviary_dim[3])
        # clipped_pos_y = np.clip(state[1], self._aviary_dim[1], self._aviary_dim[4])
        # clipped_pos_xy = np.clip(state[0:2], self._aviary_dim[0], self._aviary_dim[3])

        # clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        # clipped_pos_z = np.clip(state[2], 0, self._aviary_dim[5])
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # if self.GUI:
        #     self._clipAndNormalizeStateWarning(state,
        #                                        clipped_pos_xy,
        #                                        clipped_pos_z,
        #                                        clipped_rp,
        #                                        clipped_vel_xy,
        #                                        clipped_vel_z
        #                                        )

        # normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_xy = state[0:2] / np.array([self._aviary_dim[3], self._aviary_dim[4]])
        # normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_pos_z = state[2] / self._aviary_dim[5]

        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )
        # print("NORM AND CLIPPED", norm_and_clipped)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """
        Original PyBullet code

        Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                              state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                             state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                              state[
                                                                                                                  11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, int]
            A dict containing the current info values.
            1. The number of found targets is stored under the key "found_targets".

            Future versions may include additional info values.
        """
        return {"found_targets": self._current_target_index}

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Returns
        -------
        bool
            Whether the agent has reached the time/step limit.

        """
        if self._max_steps <= self._steps:
            return True
        return False

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # Original PyBullet termination conditions

        # print(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)
        # print(self._getDroneStateVector(0)[2] < self.COLLISION_H and self._steps > 100)
        # print(self._steps)
        #  or \self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC or

        if (self._has_collision_occurred()
                or self._is_done
                # or (self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC)
        ):
            return True
        else:
            return False

    def _computeReward(self) -> float:
        """Computes the current reward value.

        Returns
        -------
        float
            The reward value.

        """
        # Negative reward for termination
        if self._computeTerminated() and not self._is_done:
            return -100.0

        reward = np.float32(0.0)

        # try:
        #
        #     if self._current_target_index > 0:
        #         # Additional reward for progressing towards the target
        #         reward += self.calculate_progress_reward(self._current_position, self._last_position,
        #                                                  self._target_points[self._current_target_index - 1],
        #                                                  self._target_points[self._current_target_index]) * 2000
        #     else:
        #
        #         # Reward based on distance to target
        #
        #         # reward += abs(1 / self._distance_to_target + self.eps) # * self._discount ** self._steps/10
        #         reward += (np.exp(-4 * abs(self._distance_to_target))) * 3
        #         # Additional reward for progressing towards the target
        #         reward += (self._prev_distance_to_target - self._distance_to_target) * 10
        #
        #         # Negative reward for spinning too fast
        #         # reward += -np.linalg.norm(self.ang_v) / 50
        #
        #         # reward -= sum(abs (x - y) for x in self._last_action for y in self._action)
        #
        #         # Penalize large actions to avoid erratic behavior
        #         # reward -= 0.01 * np.linalg.norm(self._last_action)
        #
        # except ZeroDivisionError:
        #     # Give a high reward if the drone is at the target (avoiding division by zero)
        #     reward += 100



        # Check if the drone has reached a target
        if self._distance_to_target <= self._threshold:
            self._current_target_index += 1

            if self._current_target_index == len(self._target_points):
                # Reward for reaching all targets
                reward += 1000  # * self._discount ** self._steps/10
                self._is_done = True

            else:
                # Reward for reaching a target
                reward += 150 #* (self._discount ** (self._steps / 10))
                self.just_found = True

        else:
            # print("DISTANCE TO TARGET", self._distance_to_target)

            reward += (np.exp(-2 * abs(self._distance_to_target))) * 3
            reward += ((self._prev_distance_to_target - self._distance_to_target) * 10) if not self.just_found else 0
            self.just_found = False

        self._prev_distance_to_target = self._distance_to_target
        self._last_position = self._current_position

        return reward / 4

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

    def reset(self,
              seed: int = None,
              options: dict = None):
        """Resets the environment."""

        ret = super().reset(seed, options)

        self._is_done = False
        self._current_target_index = 0
        self._steps = 0

        # if self.random_spawn and self.total_steps < 100_000:
        #     from_p, to_p = self._target_points[np.random.choice(len(self._target_points), size=2, replace=False)]
        #     self.INIT_XYZS[0] = self.init_position_generator.generate_random_point_around_line(from_p, to_p)
        # else:
        #     # self.INIT_XYZS[0] = self.INIT_XYZS[0]
        #     self.INIT_XYZS[0] = (0,0, self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET + .1)

        self._current_position = ret[0][0:3]

        # self._steps_since_last_target = 0
        self._distance_to_target = np.linalg.norm(self._current_position - self._target_points[0])
        self._prev_distance_to_target = np.linalg.norm(self._current_position - self._target_points[0])
        self._last_action = np.zeros(4, dtype=np.float32)
        # self._reached_targets = np.zeros(len(self._target_points), dtype=bool)

        if self.GUI:
            self.show_targets()

        return ret

    def distance_between_points(self, point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def _has_collision_occurred(self) -> bool:
        """
        Checks if the drone has collided with the ground or an obstacle.

        Returns
        -------
        bool
            True if the drone has collided, False otherwise.

        """
        # (state[2] < self.COLLISION_H * 3 and self._steps > 100) or
        # COLLISION_H = 0.15  # Height at which the drone is considered to have collided with the ground.
        # Three times the collision height because the drone tend act like a "wheel"
        # and circle around a lower target point.
        # Now done with the contact detection by the PyBullet environment

        state = self.pos[0]

        if (state[0] > self._x_high or
                state[0] < self._x_low or
                state[1] > self._y_high or
                state[1] < self._y_low or
                (len(p.getContactPoints()) > 0) or
                state[2] > self._z_high):

            # print(p.getOverlappingObjects())

            return True
        else:
            return False

    def save_model(self, save_folder):
        """
        Saves the model to a folder.

        Parameters
        ----------
        save_folder : str
            The folder to save the model to.

        """
        # Get the source code of the object's class
        source_code = inspect.getsource(self.__class__)

        # Construct the file path for saving the source code
        file_path = os.path.join(save_folder, "model.py")

        # Save the source code as text
        with open(file_path, "w") as file:
            file.write(source_code)
        print(f"Object source code saved to: {file_path}")

    def distance_to_line(self, point, line_start, line_end):
        # Calculate the vector from line_start to line_end

        distance = np.linalg.norm(np.cross(line_end - line_start, point - line_start)) / np.linalg.norm(line_vector)

        return distance

    def show_targets(self):

        self.target_visual = []

        for target in self._target_points:
            self.target_visual.append(
                p.loadURDF(
                    fileName="\Sol/resources/target.urdf",
                    # "/resources/target.urdf",
                    basePosition=target,
                    useFixedBase=True,
                    globalScaling=self._threshold / 4.0,
                    physicsClientId=self.CLIENT,
                )
            )
        for i, visual in enumerate(self.target_visual):
            p.changeVisualShape(
                visual,
                linkIndex=-1,
                rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                physicsClientId=self.CLIENT
            )

    def remove_target(self):
        # delete the reached target and recolour the others
        if len(self.target_visual) > 0:
            p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

    # class NormalizeReward:
    #     r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    #
    #     The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    #
    #     Note:
    #         The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
    #         instantiated or the policy was changed recently.
    #     """
    #
    #     def __init__(
    #             self,
    #             env: gym.Env,
    #             gamma: float = 0.99,
    #             epsilon: float = 1e-8,
    #     ):
    #         """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    #
    #         Args:
    #             env (env): The environment to apply the wrapper
    #             epsilon (float): A stability parameter
    #             gamma (float): The discount factor that is used in the exponential moving average.
    #         """
    #         super().__init__(env)
    #         self.num_envs = getattr(env, "num_envs", 1)
    #         self.is_vector_env = getattr(env, "is_vector_env", False)
    #         self.return_rms = RunningMeanStd(shape=())
    #         self.returns = np.zeros(self.num_envs)
    #         self.gamma = gamma
    #         self.epsilon = epsilon
    #
    #     def step(self, action):
    #         """Steps through the environment, normalizing the rewards returned."""
    #         obs, rews, terminateds, truncateds, infos = self.env.step(action)
    #         if not self.is_vector_env:
    #             rews = np.array([rews])
    #         self.returns = self.returns * self.gamma + rews
    #         rews = self.normalize(rews)
    #         dones = np.logical_or(terminateds, truncateds)
    #         self.returns[dones] = 0.0
    #         if not self.is_vector_env:
    #             rews = rews[0]
    #         return obs, rews, terminateds, truncateds, infos
    #
    #     def normalize(self, rews):
    #         """Normalizes the rewards with the running mean rewards and their variance."""
    #         self.return_rms.update(self.returns)
    #         return rews / np.sqrt(self.return_rms.var + self.epsilon)
