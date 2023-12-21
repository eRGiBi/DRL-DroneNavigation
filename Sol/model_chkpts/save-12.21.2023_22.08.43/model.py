class PBDroneEnv(
    # BaseAviary,
    BaseSingleAgentAviary
):

    def __init__(self,
                 target_points, threshold, discount, max_steps,
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
                 ):

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         vision_attributes=vision_attributes,
                         record=record,

                         # obstacles=False,
                         # user_debug_gui=False,
                         # vision_attributes=vision_attributes,
                         )

        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self.OBS_TYPE = obs

        self._target_points = np.array(target_points)
        self._threshold = threshold
        self._discount = discount
        self._max_steps = max_steps

        self._steps = 0
        self._current_position = self.INIT_XYZS
        self._last_action = np.zeros(4)
        self._prev_distance_to_target = np.linalg.norm(self._current_position - target_points[0])
        self._current_target_index = 0
        self._is_done = False

        if save_model:
            assert save_folder is not None

            self.save_model(save_folder)

    def step(self, action):

        # The last action ended the episode. Ignore the current action and start
        # a new episode.
        # print(self._steps)

        self._steps += 1

        obs, reward, terminated, truncated, info = (
            super().step(action))

        # Create a time step
        # time_step = ts.transition(
        #     tf.convert_to_tensor(obs),
        #     tf.convert_to_tensor(reward),
        #     # discount=tf.constant(self._discount),
        # )
        self._last_action = action
        self._current_position = self._computeObs()[:3]

        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        """Returns the action space of the environment."""

        return spaces.Box(low=-1 * np.ones(4), high=np.ones(4),
                          shape=(4,), dtype=np.float64)

    def _observationSpace(self):
        """Returns the observation space of the environment."""

        return spaces.Box(low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                          high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                          dtype=np.float64
                          )

    def _computeObs(self):
        """
        Returns the current observation of the environment.

        Kinematic observation of size 12.

        """
        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        # ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        return obs.astype('float64')

    def _clipAndNormalizeState(self, state):
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

        Returns
        -------
        dict[str, int]
            A dict containing the current
            info values.
            The number of found targets is stored under the key "found_targets".

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
        # print(self._current_target_index, len(self._target_points))

        # print(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)
        # print(self._getDroneStateVector(0)[2] < self.COLLISION_H and self._steps > 100)
        # print(self._steps)
        #  or \self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC or

        if self._has_collision_occurred() or self._current_target_index == len(self._target_points):
            return True
        else:
            return False

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward value.

        """

        if self._computeTerminated() and not self._is_done:
            # print("term and NOT DONE")
            return -100 #  * np.linalg.norm(velocity)

        reward = 0.0

        distance_to_target = np.linalg.norm(
            self._target_points[self._current_target_index] - self._current_position)

        # print("tar", self._target_points[self._current_target_index])
        #
        # print("dis", distance_to_target)
        # distance_to_target = self.distance_between_points(self._computeObs()[:3],
        #                                                   self._target_points[self._current_target_index])

        # try:
            # reward -= distance_to_target ** 2
            # Reward based on distance to target

            # reward += (1 / distance_to_target)  # * self._discount ** self._steps/10

            # Additional reward for progressing towards the target
        reward += (self._prev_distance_to_target - distance_to_target) * 20

            # # Penalize large actions to avoid erratic behavior
            # reward -= 0.01 * np.linalg.norm(self._last_action)

        # except ZeroDivisionError:
        #     # Give a high reward if the drone is at the target (avoiding division by zero)
        #     reward += 20

        # Check if the drone has reached a target
        if distance_to_target <= self._threshold * 3:
            # print("IN")
            self._current_target_index += 1

            if self._current_target_index == len(self._target_points):
                # Reward for reaching all targets
                reward += 1000000.0  # * self._discount ** self._steps/10  # Reward for reaching all targets
                self._is_done = True
            else:
                # Reward for reaching a target
                reward += 10  # * (self._discount ** (self._steps / 5))

        self._prev_distance_to_target = distance_to_target
        return reward

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

    def reset(self,
              seed: int = None,
              options: dict = None):
        """Resets the environment."""

        self._is_done = False
        self._current_target_index = 0
        self._current_position = np.array([0, 0, 0])
        self._steps = 0
        self._prev_distance_to_target = np.linalg.norm(self._current_position - self._target_points[0])
        self._last_action = np.zeros(4)

        return super().reset(seed, options)

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
        # COLLISION_H = 0.15  # Height at which the drone is considered to have collided with the ground.
        # Three times the collision height because the drone tend act like a "wheel"
        # and circle around a lower target point.

        state = self._computeObs()[:3]

        if (state[0] >= 1. or state[0] <= -1. or
                state[1] >= 1. or state[1] <= -1. or
                (state[2] <= self.COLLISION_H * 3 and self._steps > 100) or state[2] >= 1):
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


    # def _computeReward(self):
    #
    #     if self._computeTerminated() and not self._is_done:
    #         # print("term and NOT DONE")
    #         return -10#  * np.linalg.norm(velocity)
    #
    #     reward = 0.0
    #     # Get the current drone position
    #     # current_position = self._computeObs()[:3]
    #
    #     distance_to_target = abs(np.linalg.norm(
    #         self._current_position - self._target_points[self._current_target_index]
    #     ))
    #
    #     # print("tar", self._target_points[self._current_target_index])
    #
    #     # print("dis", distance_to_target)
    #     # distance_to_target = self.distance_between_points(self._computeObs()[:3],
    #     #                                                   self._target_points[self._current_target_index])
    #
    #     try:
    #         # reward -= distance_to_target ** 2
    #         # Reward based on distance to target
    #         # print("dis", distance_to_target)
    #
    ##        reward += (1 / distance_to_target) * self._discount ** self._steps/10
    #
    #         # Additional reward for progressing towards the target
    #         reward += (self._prev_distance_to_target - distance_to_target) * 1.5
    #
    #         # # Penalize large actions to avoid erratic behavior
    #         reward -= 0.01 * np.linalg.norm(self._last_action)
    #
    #     except ZeroDivisionError:
    #         # Give a high reward if the drone is at the target (avoiding division by zero)
    #         reward += 10
    #
    #     # Check if the drone has reached a target
    #     if distance_to_target <= self._threshold:
    #         # print("IN")
    #         self._current_target_index += 1
    #
    #         if self._current_target_index == len(self._target_points):
    #             # Reward for reaching all targets
    #             reward += 100000.0  * self._discount ** self._steps/10  # Reward for reaching all targets
    #             self._is_done = True
    #         else:
    #             # Reward for reaching a target
    #             reward += 1000 * self._discount ** self._steps / 10
    #
    #         # If the drone is outside the threshold, give a reward based on distance
    #     #            reward = max(0.0, 1.0 - distance_to_target / self._threshold)
    #
    #     # if self._computeTerminated() and not self._is_done:
    #     #     reward -= 1
    #     #
    #     # if (np.linalg.norm(self._computeObs()[:3] - self._target_points[self._current_target_index])) < self._threshold:
    #     #     self._current_target_index += 1
    #     #     if self._current_target_index == len(self._target_points):
    #     #         self._is_done = True
    #     #         reward += 10
    #     #     else:
    #     #         reward += 1
    #     self._prev_distance_to_target = distance_to_target
    #     return reward
