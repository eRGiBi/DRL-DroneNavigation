class PBDroneEnv(
    # BaseAviary,
    # FlyThruGateAviary,
    BaseSingleAgentAviary,
):

    def __init__(self,
                 target_points, threshold, discount, max_steps, aviary_dim,
                 save_folder=None,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.THRUST,
                 vision_attributes=False,
                 user_debug_gui=False,
                 obstacles=False,
                 random_spawn=False,
                 cylinder=True,
                 circle=False,
                 include_target=False,
                 include_distance=False,
                 normalize_actions=False,
                 ):

        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self.OBS_TYPE = obs
        self._OBS_TYPE = obs

        self._target_points = np.array(target_points)
        # self._target_points = np.append(self._target_points, target_points)
        self._or_target_points = np.array(target_points)
        self._reached_targets = np.zeros(len(self._target_points), dtype=bool)

        self._threshold = threshold
        self._discount = discount
        self._max_steps = max_steps

        self.initial_xyzs = initial_xyzs
        self._aviary_dim = aviary_dim
        self._x_low, self._y_low, self._z_low, self._x_high, self._y_high, self._z_high = aviary_dim
        self.circle_radius = 1
        self.cylinder = cylinder
        self.circle = circle

        self.include_target = include_target
        self.obs_goal_horizon = 1
        self.include_distance = include_distance
        self._max_target_dist = max(abs(self._x_low) + self._x_high, abs(self._y_low) + self._y_high, self._z_high)

        self.random_spawn = random_spawn
        self.initial_state_buffer = []

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
                         # act=ActionType.RPM
                         )

        # if self.ACT_TYPE == ActionType.THRUST:
        a_low = self.KF * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST) ** 2
        a_high = self.KF * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST) ** 2
        self.physical_action_bounds = (np.full(4, a_low, np.float32),
                                       np.full(4, a_high, np.float32))
        self.normalize_actions = normalize_actions

        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        self._current_position = self.INIT_XYZS[0]
        # self._last_position = self._current_position

        self.current_vel, self.current_ang_v = np.zeros(3), np.zeros(3)
        self.prev_vel = np.zeros(3)  # Previous linear velocities
        self.prev_ang_v = np.zeros(3)  # Previous angular velocities

        # self.steps_since_last_target = 0

        self._last_action = np.zeros(4, dtype=np.float32)

        # Smallest possible value for the data type to avoid underflow exceptions
        self.eps = np.finfo(self._last_action.dtype).eps
        self.eps = np.finfo(self._last_action.dtype).tiny

        self._distance_to_target = np.linalg.norm(self._current_position - target_points[0])
        self._prev_distance_to_target = np.linalg.norm(self._current_position - target_points[0])
        self._current_target_index = 0
        self.just_found = False

        self._is_done = False

        self._steps = 0
        self.total_steps = 0

        # Saving
        self.save_folder = save_folder
        self.rollout_path = os.path.join('Sol/rollouts/', 'rollout_' +
                                         str(sum(len(files) for _, _, files in os.walk('Sol/rollouts/')) + 1) + '.txt')
        self.lock = threading.Lock()

        if save_folder is not None:
            self.save_model(save_folder)

        # Visualization
        self.CLIENT = self.getPyBulletClient()
        self.target_visual = []
        if gui:
            self.show_targets()

        # self._addObstacles()

        if self.random_spawn:
            self.PositionGenerator = PositionGenerator(self._aviary_dim, 0.1)

    def step(self, action):
        """Applies the given action to the environment."""

        if self.normalize_actions:
            obs, reward, terminated, truncated, info = (
                super().step(self.rescale_action(action))
            )
        else:
            obs, reward, terminated, truncated, info = (
                super().step(action)
            )
        # print(self.getDroneLookDirection(0))
        # print(self.get_forward_vector(), self.gasd())
        # print(self.orientation_reward(self._target_points[self._current_target_index]))
        # self._showDroneLocalAxes(0)

        # print("ACTION", action)
        # print("OBS", obs)
        # print("pos", self.pos[0])
        # print("REWARD", reward)
        # print("TERMINATED", terminated)

        # Rollout collection
        # self.collect_rollout(obs, reward)

        if not terminated:
            self.update_state_post_step(action)

        return obs, reward, terminated, truncated, info

    def update_state_post_step(self, action):
        self._steps += 1
        # self.total_steps += 1
        self._last_action = action
        # self._last_position = copy.deepcopy(self._current_position)
        self._current_position = deepcopy(self.pos[0])  # + self.eps

        self.prev_vel, self.prev_ang_v = deepcopy(self.current_vel), deepcopy(self.current_ang_v)
        self.current_vel, self.current_ang_v = deepcopy(self.vel[0]), deepcopy(self.ang_v[0])

        # Calculate the Euclidean distance between the drone and the next target
        self._distance_to_target = np.linalg.norm(
            self.current_target() - self._current_position)

        # distance_to_target = self.distance_between_points(self._computeObs()[:3],
        #                                                   self._target_points[self._current_target_index])

        if (self.GUI and self._distance_to_target <= self._threshold \
                # and not (self.random_spawn and self.total_steps < 100_000)
        ):
            self.remove_target()

    def _actionSpace(self):
        """Returns the action space of the environment."""

        # return super()._actionSpace()

        #or self.ACT_TYPE == ActionType.RPM
        if self.normalize_actions:
            return spaces.Box(low=-1 * np.ones(4, dtype=np.float32),
                              high=np.ones(4, dtype=np.float32),
                              shape=(4,), dtype=np.float32)

        else:
            # if self.ACT_TYPE == ActionType.THRUST:
            return spaces.Box(low=self.physical_action_bounds[0],
                              high=self.physical_action_bounds[1],
                              dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""

        # self.x_threshold = 2
        # self.y_threshold = 2
        # self.z_threshold = 2
        # self.phi_threshold_radians = 85 * math.pi / 180
        # self.theta_threshold_radians = 85 * math.pi / 180
        # self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.
        #
        #
        # # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        # low = np.array([
        #     -self.x_threshold, -np.finfo(np.float32).max,
        #     -self.y_threshold, -np.finfo(np.float32).max,
        #     self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
        #     -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
        #     -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
        # ])
        # high = np.array([
        #     self.x_threshold, np.finfo(np.float32).max,
        #     self.y_threshold, np.finfo(np.float32).max,
        #     self.z_threshold, np.finfo(np.float32).max,
        #     self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
        #     np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
        # ])

        low = np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

        # if self.include_target:
        #     # Include future goal state(s)
        #     mul = 1 + self.obs_goal_horizon
        #     low = np.concatenate([low] * mul)
        #     high = np.concatenate([high] * mul)

        if self.include_distance:
            low = np.append(low, 0)
            high = np.append(high, 1)

        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        return observation_space

        # Without normalization
        # return spaces.Box(low=np.array([self._x_low, self._y_low, 0,
        #                                 -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        #                   high=np.array([self._x_high, self._y_high, self._z_high,
        #                                  1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        #                   dtype=np.float32
        #                     )

    def _computeObs(self):
        """
        Returns the current observation of the environment.

        Kinematic observation of size 12 or 13.

        """
        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))

        # obs = obs + self.eps
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )

        if self.include_distance:
            ret = np.append(ret, [self._distance_to_target / self._max_target_dist])

            # ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
            # for i in range(1, self.obs_goal_horizon + 1):
            #     future_target = self._target_points[self._current_target_index + i]
            #     future_target = self._clipAndNormalizeState(future_target)
            #     ret = np.hstack([ret, future_target[0:3], future_target[7:10], future_target[10:13],
            #     future_target[13:16]]).reshape(12 * (i + 1), )

        # print("x", obs[0], "y", obs[1], "z", obs[2])
        # print("roll", obs[7], "pitch", obs[8], "yaw", obs[9])
        # print("vel_x", obs[10], "vel_y", obs[11], "vel_z", obs[12])
        # print("ang_vel_x", obs[13], "ang_vel_y", obs[14], "ang_vel_z", obs[15])

        # ret = np.array([ret[0]/self._x_high, ret[1]/self._y_high, ret[2]/self._z_high,
        #                 ret[3], ret[4], ret[5], ret[6], ret[7], ret[8], ret[9], ret[10], ret[11]])

        try:
            # Normalize observations to avoid underflow/overflow issues, STILL doesn't work
            ret = np.clip(ret, np.finfo(np.float32).min, np.finfo(np.float32).max)

            ret = ret.astype(np.float32)

            assert isinstance(ret, np.ndarray) and ret.dtype == np.float32, \
                "The observation is not a numpy array of type float32"

            return ret
        except FloatingPointError as e:
            print("Error in _computeObs():", ret)
            print(f"Underflow error: {e}")
            # Return a clipped array to handle underflow
            return np.clip(ret, np.finfo(np.float32).min, np.finfo(np.float32).max)

    def _clipAndNormalizeState(self, state):
        """
        Original PyBullet code modified for the new observation spaces.
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
        Original PyBullet code, unused

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

        # Original PyBullet termination conditions:
        # print(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)
        # print(self._getDroneStateVector(0)[2] < self.COLLISION_H and self._steps > 100)

        if (self._is_done or
                self._has_collision_occurred()
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

        # rew = self.progress_reward()
        # self._prev_distance_to_target = self._distance_to_target
        # self._last_position = copy.deepcopy(self._current_position)
        # return rew

        # Negative reward for termination
        if self._computeTerminated() and not self._is_done:
            return -10.0

        reward = np.float32(0.0)

        # Random spawning
        # if self.random_spawn and self.total_steps < 500_000:
        #     min_dis = 0
        #
        #     if self._current_target_index == len(self._target_points):
        #         return 10
        #
        #     for i, target in enumerate(self._target_points):
        #         dis = np.linalg.norm(self._current_position - target)
        #         if dis < self._threshold and not self._reached_targets[i]:
        #             self.remove_target(i)
        #             # self.remove_target(target, i)
        #             self._reached_targets[i] = True
        #             return 3
        #         elif min_dis == 0 or dis < min_dis:
        #             min_dis = dis
        #             self._current_target_index = i
        #             self._distance_to_target = min_dis
        #
        #     self._last_position = deepcopy(self._current_position)
        #
        #     reward += (np.exp(-2 * self._distance_to_target)) * 3
        #     reward += ((self._prev_distance_to_target - self._distance_to_target) * 3000) if not self.just_found else 0
        #     reward += self.orientation_reward(self.current_target()) * 3
        #     reward += self.smoothness_reward()
        #
        #     self.just_found = False
        #
        #     return reward / 25

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

        # Check if the drone has reached a target
        if self._distance_to_target <= self._threshold:
            self._current_target_index += 1

            if self._current_target_index == len(self._target_points):
                # Reward for reaching all targets
                reward += 200
                # reward += self.smoothness_reward(0.9, 0.9)
                self._is_done = True

            else:
                # Reward for reaching a target
                reward += 75
                reward += self.orientation_reward(self.current_target()) * 5
                self.just_found = True

        else:
            reward += (np.exp(-2 * self._distance_to_target)) * 3
            reward += ((self._prev_distance_to_target - self._distance_to_target) * 3000) if not self.just_found else 0
            reward += self.orientation_reward(self.current_target()) * 3
            reward += self.smoothness_reward()

            # print("smoothness", self.smoothness_reward())
            # print("ori", self.orientation_reward(self._target_points[self._current_target_index])* 3)
            # print("prev dis", ((self._prev_distance_to_target - self._distance_to_target) * 3000) if not self.just_found else 0)
            # print(self._prev_distance_to_target , self._distance_to_target)
            # print("exp", (np.exp(-2 * self._distance_to_target)) * 3)

            self.just_found = False

        self._prev_distance_to_target = deepcopy(self._distance_to_target)
        # self._last_position = deepcopy(self._current_position)

        return reward / 25

    def orientation_reward(self, target_pos):
        threshold_angle = np.radians(10)

        forward_vector = self.get_forward_vector()
        drone_to_target_vector = np.array(target_pos) - np.array(self.pos[0])
        drone_to_target_vector /= np.linalg.norm(drone_to_target_vector)  # Normalize

        # angle between the drone's forward vector and the vector to the target
        angle = np.arccos(np.clip(np.dot(forward_vector, drone_to_target_vector), -1.0, 1.0))

        if angle > threshold_angle:
            return -1  # Negative reward
        else:
            return 0  # No penalty

    def get_forward_vector(self):
        euler = self.rpy[0]
        # Assuming the drone's forward vector points along the x-axis in its local frame
        forward_vector = np.array([
            np.cos(euler[2]) * np.cos(euler[1]),  # Cos(yaw) * Cos(pitch)
            np.sin(euler[2]) * np.cos(euler[1]),  # Sin(yaw) * Cos(pitch)
            np.sin(euler[1])  # Sin(pitch)
        ])

        return forward_vector

    def smoothness_reward(self, accel_threshold=0.7, ang_accel_threshold=0.3):
        """
        Penalizes the drone for sudden changes in velocity or angular velocity.
        """
        lin_acc = np.linalg.norm(self.current_vel - self.prev_vel)
        ang_acc = np.linalg.norm(self.current_ang_v - self.prev_ang_v)

        linear_penalty = -abs(lin_acc) if lin_acc > accel_threshold else 0
        angular_penalty = -abs(ang_acc) if ang_acc > ang_accel_threshold else 0

        return linear_penalty + angular_penalty

    def reset(self,
              seed: int = None,
              options: dict = None):
        """Resets the environment."""

        ret = super().reset(seed, options)

        self._is_done = False
        self._current_target_index = 0
        self._steps = 0
        self._target_points = self._or_target_points

        # if self.random_spawn and self.total_steps < 300_000:
        #     from_p, to_p = self._target_points[np.random.choice(len(self._target_points), size=2, replace=False)]
        #     self.INIT_XYZS[0] = self.PositionGenerator.generate_random_point_around_line(from_p, to_p)
        #     # print(from_p, to_p )
        #     # print(self.INIT_XYZS[0])
        #
        #     self._current_position = self.INIT_XYZS[0]
        #     self._current_target_index = 0
        # else:
        #     self.INIT_XYZS[0] = self.initial_xyzs
        #     # self.INIT_XYZS[0] = self.INIT_XYZS[0]
        #     # self.INIT_XYZS[0] = (0,0, self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET + .1)
        #     # self._current_position = ret[0][0:3]

        # if len(self.initial_state_buffer) > 0:
        #     self.INIT_XYZS[0] = self.initial_state_buffer[np.random.randint(len(self.initial_state_buffer))]
        # else:
        # segment_index = np.random.randint(len(self._target_points) - 1)
        # segment_center = (self._target_points[segment_index] + self._target_points[segment_index + 1]) / 2
        # self.INIT_XYZS[0] = segment_center
        # shifted_target_points = np.concatenate(
        #     (self._target_points[segment_index + 1:], self._target_points[:segment_index + 1]), axis=0)
        #
        # self._target_points = shifted_target_points

        # self._steps_since_last_target = 0
        self._distance_to_target = np.linalg.norm(self._current_position - self._target_points[0])
        self._prev_distance_to_target = np.linalg.norm(self._current_position - self._target_points[0])
        self.prev_vel = np.zeros(3)
        self.prev_ang_v = np.zeros(3)
        # self._last_action = np.zeros(4, dtype=np.float32)
        # self._last_position = copy.deepcopy(self._current_position)
        self.just_found = False

        self._reached_targets = np.zeros(len(self._target_points), dtype=bool)

        if self.GUI:
            self.show_targets()

        return ret

    def shift_targets(self, shift):
        self._target_points = np.roll(self._target_points, shift, axis=0)
        # self._or_target_points = np.roll(self._or_target_points, shift, axis=0)

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
        # If initialized on the ground: (state[2] < self.COLLISION_H * 3 and self._steps > 100) or
        # COLLISION_H = 0.15 # Height at which the drone is considered to have collided with the ground.
        # Three times the collision height because the drone tends to act like a "wheel"
        # Now done with the contact detection by the PyBullet environment

        state = self.pos[0]

        if (state[0] > self._x_high or
                state[0] < self._x_low or
                state[1] > self._y_high or
                state[1] < self._y_low or
                (len(p.getContactPoints()) > 0) or
                state[2] > self._z_high or
                (self.cylinder and self.is_out_of_cylinder_bounds(state))
        ):
            return True
            # print(p.getOverlappingObjects())

        else:
            return False

    def current_target(self):
        return self._target_points[self._current_target_index] if self._current_target_index < len(
            self._target_points) else None

    def next_target(self):
        return self._target_points[self._current_target_index + 1] if (
                self._current_target_index + 1 != len(self._target_points)) \
            else None

    def is_out_of_cylinder_bounds(self, drone_position, circle_center=(0, 0, 1), extension_length=0.2):
        """
        Compute the closest point on the circle to a given position,
        check if the drone is out of the specified bounds from the nearest point on a circle.
        """
        if self.circle:

            drone_vec = np.array(drone_position)
            center_vec = np.array(circle_center)

            # Vector from center to drone
            center_to_drone_vec = drone_vec - center_vec

            # Ignore z-coordinate for XY plane circle
            center_to_drone_vec[2] = 0

            # Normalize and scale to circle radius
            try:
                norm_vec = center_to_drone_vec / np.linalg.norm(center_to_drone_vec) * self.circle_radius
            except FloatingPointError:
                norm_vec = np.zeros_like(center_to_drone_vec)
            closest_point = center_vec + norm_vec

            distance_from_closest_point = np.linalg.norm(np.array(drone_position) - closest_point)

            return distance_from_closest_point > self._threshold

        else:
            # return False

            if self._current_target_index == 0:
                base1 = np.array(self.INIT_XYZS[0])
                base2 = np.array(self.current_target())
            else:
                base1 = np.array(self._target_points[self._current_target_index - 1])
                base2 = np.array(self.current_target())

            # Vector from point1 to point2
            line_vec = base2 - base1
            line_length = np.linalg.norm(line_vec)

            if line_length == 0:
                return np.linalg.norm(drone_position - base1) > self._threshold

            line_unit_vec = line_vec / line_length

            # # Vector from point1 to drone
            # point1_to_drone_vec = drone_position - base1
            #
            # # Projection of drone vector onto the line segment
            # projection_length = np.dot(point1_to_drone_vec, line_unit_vec)
            #
            # # Clamp the projection length to the bounds of the line segment
            # projection_length = np.clip(projection_length, 0, line_length)
            #
            # # Closest point on the line segment
            # closest_point_on_line = base1 + projection_length * line_unit_vec
            #
            # # Distance from the drone to the closest point on the line
            # distance_from_line = np.linalg.norm(drone_position - closest_point_on_line)
            #
            # return distance_from_line > self._threshold

            # Extend the line segment by extension_length on both ends
            extended_point1 = base1 - extension_length * line_unit_vec
            extended_point2 = base2 + extension_length * line_unit_vec

            # Vector from extended_point1 to drone
            point1_to_drone_vec = drone_position - extended_point1

            # Projection of drone vector onto the extended line segment
            projection_length = np.dot(point1_to_drone_vec, line_unit_vec)

            # Clamp the projection length to the bounds of the extended line segment
            projection_length = np.clip(projection_length, 0, np.linalg.norm(extended_point2 - extended_point1))

            # Closest point on the extended line segment
            closest_point_on_line = extended_point1 + projection_length * line_unit_vec

            # Distance from the drone to the closest point on the line
            distance_from_line = np.linalg.norm(drone_position - closest_point_on_line)

            return distance_from_line > self._threshold + extension_length

    def save_model(self, save_folder):
        """
        Saves the model to a folder.

        Parameters
        ----------
        save_folder : str
            The folder to save the model to.
        """
        source_code = inspect.getsource(self.__class__)

        file_path = os.path.join(save_folder, "model.py")

        with open(file_path, "w") as file:
            file.write(source_code)
        print(f"Object source code saved to: {file_path}")

    def distance_to_line(self, point, line_start, line_end):

        distance = np.linalg.norm(np.cross(line_end - line_start, point - line_start)) / np.linalg.norm(line_vector)

        return distance

    def collect_rollout(self, obs, reward):
        if len(obs) > 0:
            with open(self.rollout_path, mode='a+') as f:
                with self.lock:  # Doesnt work even with thread locking with multiple envs
                    for x in obs.tolist():
                        f.write(str(np.format_float_positional(np.float32(x), unique=False, precision=32)) + ",")
                    f.write(str(reward))
                    f.write("\n")
            f.close()

    def show_targets(self):
        """Visualizes the targets in PyBullet."""

        self.target_visual = []
        # for _ in range(len(self.target_visual)):
        #     p.removeBody()

        for target, reached in zip(self._target_points, self._reached_targets):
            self.target_visual.append(
                p.loadURDF(
                    fileName=os.path.normpath("./Sol/resources/target.urdf"),
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

    def remove_target(self, index=None):
        """Removes the target from the PyBullet visualization."""
        if len(self.target_visual) > 0:
            p.removeBody(self.target_visual[0]) if index is None else p.removeBody(self.target_visual[index])
            # if target is None:
            if not self.random_spawn:
                self.target_visual.pop(0)
            elif self.total_steps > 100_000 and index is not None:
                self.target_visual.pop(index)

    def _preprocessAction(self, action):
        """Converts the action passed to .step() into motors' RPMs (ndarray of shape (4,)).
        From safe-control-gym.

        Args:
            action (ndarray): The raw action input, of size depending on QUAD_TYPE.

        Returns:
            action (ndarray): The motors RPMs to apply to the quadrotor.
        """

        # if self.ACT_TYPE == ActionType.THRUST:
        # action = self.denormalize_action(action)
        # self.current_physical_action = action

        # action = action = (1 + self.norm_act_scale * action) * self.hover_thrust

        thrust = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        # self.current_clipped_action = thrust

        # convert to quad motor rpm commands
        pwm = cmd2pwm(thrust, self.PWM2RPM_SCALE, self.PWM2RPM_CONST, self.KF, self.MIN_PWM, self.MAX_PWM)
        rpm = pwm2rpm(pwm, self.PWM2RPM_SCALE, self.PWM2RPM_CONST)
        return rpm

        # elif self.ACT_TYPE == ActionType.RPM:
        #     # print("action: ", action)
        #     # print(np.array(self.HOVER_RPM * (1+0.05*action)))
        #     return np.array(self.HOVER_RPM * (1 + 0.05 * action))
        # elif self.ACT_TYPE == ActionType.PID:
        #     state = self._getDroneStateVector(0)
        #     next_pos = self._calculateNextStep(
        #         current_position=state[0:3],
        #         destination=action,
        #         step_size=1,
        #     )
        #     rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
        #                                          cur_pos=state[0:3],
        #                                          cur_quat=state[3:7],
        #                                          cur_vel=state[10:13],
        #                                          cur_ang_vel=state[13:16],
        #                                          target_pos=next_pos
        #                                          )
        #     return rpm

    def normalize_action(self, action):
        """Converts a physical action into an normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        """
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = (action / self.hover_thrust - 1) / self.norm_act_scale

        return action

    def denormalize_action(self, action):
        """Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        """
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = (1 + self.norm_act_scale * action) * self.hover_thrust

        return action

    def rescale_action(self, action):
        min_action = self.physical_action_bounds[0]
        max_action = self.physical_action_bounds[1]
        assert isinstance(self.action_space, spaces.Box), f"expected Box action space, got {type(self.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        min_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + min_action
        max_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype) + max_action

        # assert np.all(np.greater_equal(action, min_action)), (action, min_action)
        # assert np.all(np.less_equal(action, max_action)), (action, max_action)
        low = self.action_space.low
        high = self.action_space.high
        action = low + (high - low) * ((action - min_action) / (max_action - min_action))
        action = np.clip(action, low, high)

        return action
