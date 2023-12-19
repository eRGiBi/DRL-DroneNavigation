import time

import pybullet as p
import numpy as np


class DroneEnv:
    def __init__(self):
        # Initialize PyBullet
        p.connect(p.GUI)  # You can use p.DIRECT for headless mode

        # Load the drone and set up the environment
        self.drone_urdf_path = "resources/cf2p.urdf"
        self.drone_id = p.loadURDF(self.drone_urdf_path, [0, 0, 1])

        # Define points in a sequence that the drone should reach
        self.target_points = [
            [1, 0, 1],
            [20, 0, 10],
            [50, 20, 10],
            # Add more points as needed
        ]
        self.current_target_index = 0

        # Define reward parameters
        self.reward_reaching_target = 10.0
        self.reward_distance_penalty = -0.1

    def reset(self):
        # Reset the environment for a new episode
        p.resetSimulation()
        self.drone_id = p.loadURDF(self.drone_urdf_path, [0, 0, 1])
        self.current_target_index = 0
        return self.get_observation()

    def get_observation(self):
        # Get observation (e.g., drone position, velocity, etc.)
        # Modify this based on your specific observation space
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        return np.array(drone_pos)

    def step(self, action):
        # Take a step in the environment based on the given action
        # Modify this based on your specific action space
        p.applyExternalForce(self.drone_id, -1, action, [0, 0, 0], p.LINK_FRAME)

        # Calculate reward
        reward = self.calculate_reward()

        # Check if the drone has reached the current target
        if self.has_reached_target():
            self.current_target_index += 1

        # Check if all targets have been reached
        done = self.current_target_index == len(self.target_points)

        # Return the next observation, reward, and done flag
        return self.get_observation(), reward, done

    def calculate_reward(self):
        # Calculate the reward based on the distance to the current target
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        target_pos = self.target_points[self.current_target_index]
        distance = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
        reward = self.reward_reaching_target - self.reward_distance_penalty * distance
        return reward

    def has_reached_target(self):
        # Check if the drone has reached the current target
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        target_pos = self.target_points[self.current_target_index]
        distance = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
        return distance < 0.1  # Adjust the threshold as needed

    def close(self):
        # Close the PyBullet simulation
        p.disconnect()


# Example usage:
env = DroneEnv()
observation = env.reset()

for _ in range(100000):
    # Replace this with your RL algorithm to select actions
    action = np.random.uniform(-1, 1, size=(3,))

    # Take a step in the environment
    observation, reward, done = env.step(action)


    if done:
        observation = env.reset()

while True:
    time.sleep(1)
env.close()
while True:
    time.sleep(1)
