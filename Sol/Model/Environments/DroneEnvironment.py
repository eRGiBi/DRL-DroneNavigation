import os

import gym
from gymnasium import spaces
import numpy as np
import pybullet
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from PyBullet.BaseAviary import BaseAviary
from PyBullet.enums import DroneModel, Physics, ImageType, ActionType, ObservationType


class DroneEnvironment(
    BaseAviary,
    # py_environment.PyEnvironment,
):

    def __init__(self,
                 race_track, target_points, threshold, drone, discount=1,
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

                 ):

        super(DroneEnvironment, self).__init__(drone_model=drone_model,
                                               num_drones=1,
                                               initial_xyzs=initial_xyzs,
                                               initial_rpys=initial_rpys,
                                               physics=physics,
                                               pyb_freq=pyb_freq,
                                               ctrl_freq=ctrl_freq,
                                               gui=gui,
                                               record=record,
                                               obstacles=False,
                                               user_debug_gui=False,
                                               vision_attributes=vision_attributes,
                                               )
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self.OBS_TYPE = obs
        self._gui = gui

        self._drone = drone
        self._race_track = race_track
        self._target_points = np.array(target_points)
        self._threshold = threshold
        self._discount = discount

        self._current_position = np.array([0.0, 0.0, 0.0])
        self._current_target_index = 0
        self._current_target = target_points[self._current_target_index]
        self._is_done = False

        # Possible actions: the 4 rotor speed
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,),  # Rotor speeds for 4 rotors
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='action'
        )

        # The observation: what state are we in: where is the drone and where is our next goal
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,),  # [x, y, z, target_x, target_y, target_z]
            dtype=np.float32,
            # minimum=[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            minimum=[0, 0, 0, 0, 0, 0],
            maximum=[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            name='observation'
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_position = np.array([0.0, 0.0, 0.0])
        self._current_target = self._target_points[0]
        self._is_done = False
        # return ts.restart(tf.convert_to_tensor(self._get_observation()))
        return self.reset()

    def _step(self, action):
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
        if self._is_done:
            return self.reset()

        super(BaseAviary).step(self._preprocessAction(action))

        # # Update position based on action
        # self._current_position += action

        # Update position based on action
        self._update_position(self._preprocessAction(action))

        # Calculate distance to the current target
        distance_to_target = np.linalg.norm(self._current_position - self._target_points[self._current_target])

        # Reward based on distance to target
        reward = -distance_to_target

        # Check if the drone has reached the target
        if distance_to_target < self._threshold:
            reward += 10
            self._current_target_index += 1

            if self._current_target_index == len(self._target_points):
                self._is_done = True
                reward += 100.0  # Reward for reaching all targets
            else:
                self._current_target += self._target_points[self._current_target_index]

        # Create a time step
        # termination?????????????
        time_step = ts.transition(
            tf.convert_to_tensor(self._get_observation()),
            tf.convert_to_tensor(reward),
            discount=tf.constant(self._discount),
            #  step_type=tf.convert_to_tensor(1 if self._is_done else 0)
        )

        return time_step

    def _get_observation(self):
        target_position = self._target_points[self._current_target_index] if self._current_target < len(
            self._target_points) else [0.0, 0.0, 0.0]
        return np.concatenate([self._current_position, target_position])

    def _update_position(self, rotor_speeds):
        # Implement the dynamics of the drone based on rotor speeds
        # Update the position based on the dynamics

        self._race_track.applyExternalForce(self._drone, -1, rotor_speeds, pybullet.LINK_FRAME)
        self._current_position = self._race_track.getBasePositionAndOrientation(self._drone)

    def _actionSpace(self):
        """Returns the action space of the environment."""

        return spaces.Box(low=-1 * np.ones(4), high=np.ones(4),
                          shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""

        return spaces.Box(low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                          high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                          dtype=np.float32
                          )

    def _computeObs(self):
        """Returns the current observation of the environment."""

        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        return ret.astype('float32')

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1
        self.EPISODE_LEN_SEC = 1
        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
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

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
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
        """Debugging printouts associated to `_clipAndNormalizeState`.

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

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        return False

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        thrust = self.HOVER_RPM * (1 + 0.05 * action)
        print(thrust)

        # Convert thrust to RPM for all motors
        # rpm = np.array([thrust] * 4)

        return thrust

        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1 + 0.05 * action))

        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3],  # same as the current position
                                                 target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                 target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vector
                                                 # target the desired velocity vector
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1 + 0.05 * action), 4)
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")


    def _computeReward(self):
        thresh_dist = 7
        beta = 1
        reward = 0
        z = -10
        # distance_to_target = np.linalg.norm(
            # self._current_position - self._target_points[self._current_target_index])
        print("pos", self._getDroneStateVector(0))

        distance_to_target = np.linalg.norm(
            self._getDroneStateVector(0)[:3]  - self._target_points[self._current_target_index])
        print(distance_to_target)
        reward -= distance_to_target

        # Check if the drone has reached the target
        if distance_to_target < self._threshold:
            reward += 10
            self._current_target_index += 1
            if self._current_target_index == len(self._target_points):
                self._is_done = True
                reward += 100.0  # Reward for reaching all targets
            else:
                self._current_target += self._target_points[self._current_target_index]

        else:
            # If the drone is outside the threshold, give a reward based on distance
            reward = max(0.0, 1.0 - distance_to_target / self._threshold)
        print("rew: ", reward, self._discount, self.step_counter)
        return reward * self._discount ** self.step_counter

    # quad_pt = np.array(
    #     list((self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val,)))
    #
    # if self.state["collision"]:
    #     reward = -100
    # else:
    #     dist = 10000000
    #     for i in range(0, len(pts) - 1):
    #         dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(
    #             pts[i] - pts[i + 1]))
    #
    #     if dist > thresh_dist:
    #         reward = -10
    #     else:
    #         reward_dist = math.exp(-beta * dist) - 0.5
    #         reward_speed = (np.linalg.norm(
    #             [self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val, ]) - 0.5)
    #         reward = reward_dist + reward_speed
    #
    # def interpret_action(self, action):
    #     if action == 0:
    #         quad_offset = (self.step_length, 0, 0)
    #     elif action == 1:
    #         quad_offset = (0, self.step_length, 0)
    #     elif action == 2:
    #         quad_offset = (0, 0, self.step_length)
    #     elif action == 3:
    #         quad_offset = (-self.step_length, 0, 0)
    #     elif action == 4:
    #         quad_offset = (0, -self.step_length, 0)
    #     elif action == 5:
    #         quad_offset = (0, 0, -self.step_length)
    #     else:
    #         quad_offset = (0, 0, 0)
