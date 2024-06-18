import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Sol.Utilities.Plotter as Plotter


class Track:
    """
    Initialized with a track function.

    Attributes are the waypoints, init pos and dim.
    """
    def __init__(self, track, circle=False):
        self.waypoints, self.initial_xyzs, self.aviary_dim = track
        self.is_circle = circle

    def __str__(self):
        return (f"Track with {len(self.waypoints)} waypoints, "
                f"initial position {self.initial_xyzs}, and aviary dimensions {self.aviary_dim}")


def normalize_coordinates(coordinates, new_size):
    """
    Normalize the coordinates to fit within a new size range.

    Args:
        coordinates (np.array): Original coordinates as a numpy array of shape (n, 3).
        new_size (float): The desired new size for normalization.

    Returns:
        np.array: Normalized coordinates.
    """
    # Get the min and max of the current coordinates
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)

    # Calculate the current range
    current_range = max_coords - min_coords

    # Calculate the scaling factor
    scaling_factor = new_size / current_range

    # Normalize the coordinates
    normalized_coords = (coordinates - min_coords) * scaling_factor

    return normalized_coords


def parametric_eq(num_points=5):
    """Parametric equations for a racetrack."""

    # Define the number of track points
    # Create a smooth racetrack using sine and cosine functions
    theta = np.linspace(0, 2 * np.pi, num_points)
    radius = 5.0  # Can be adjusted for a larger/smaller track
    # Parametric equations for x, y, and z coordinates
    x = radius * np.cos(theta)
    y = radius * np.cos(theta)
    z = 0.1 * np.sin(1 * theta)  # Adjust the amplitude and frequency for a smoother track

    return [np.array([x[i], y[i], z[i]]) for i in range(num_points)]


def up():
    return [
        np.array([0.0, 0.0, 0.1]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.7]),
        np.array([0.0, 0.0, 1]),
    ], np.array([0.0, 0.0, 0.1]), np.array([-2, -2, 0, 2, 2, 2])


def half_up_forward():
    return [
        np.array([0., 0., 0.5]),
        np.array([0., 0., 1]),
        np.array([0., 1, 1.5]),
        # np.array([0.5, 1.5, 1.5]),
        # np.array([1.5, 1.5, 1.5]),
    ], np.array([0., 0., 0.1]), np.array([-2, -2, 0, 2, 2, 2])


def up_circle():
    return [
        np.array([0.0, 0.0, 0.2]),
        np.array([0.1, 0.0, 0.3]),
        np.array([0.1, 0.2, 0.7]),
        np.array([0.3, 0.5, 1.5]),
        np.array([0.5, 1., 1.5]),
        np.array([1., 1., 1.5]),
        np.array([1.5, 1., 1.5]),
        np.array([1.5, 1.5, 1]),
        np.array([1.5, 0.5, 1]),
        np.array([1, 0.5, 0.5]),
        np.array([0.5, 0.2, 0.2]),
        np.array([0.0, 0.0, 0.2]),
    ], np.array([[0.0, 0.0, 0.1]]), np.array([-2, -2, 0, 2, 2, 2])


def up_sharp_back_turn():
    return [
        np.array([0.0, 0.0, 0.5]),
        np.array([-0.5, 0.2, 0.7]),
        np.array([0.3, 0.5, 0.7]),
        np.array([1, 0.5, 1]),
        np.array([1.5, 1., 1.2])
    ], np.array([[0.0, 0.0, 0.1]]), np.array([-2, -2, 0, 2, 2, 2])


def circle(radius, num_points, height, center=(0, 0, 0), plane="XY"):
    """
        Generate points on a circle in 3D space.

        Parameters:
        - radius (float): Radius of the circle.
        - num_points (int): Number of points to generate.
        - center (tuple): 3D center of the circle (x, y, z).
        - plane (str): The plane in which the circle lies, options are "XY", "XZ", "YZ".

        Returns:
        - np.array: Array of shape (num_points, 3) containing 3D coordinates.
        """

    angles = np.linspace(0, 2 * np.pi, num_points + 1, endpoint=True)
    circle_points = np.zeros((num_points + 1, 3))

    if plane == "XY":
        circle_points[:, 0] = center[0] + radius * np.cos(angles)  # X coordinate
        circle_points[:, 1] = center[1] + radius * np.sin(angles)  # Y coordinate
        circle_points[:, 2] = center[2] + height
    elif plane == "XZ":
        circle_points[:, 0] = center[0] + radius * np.cos(angles)  # X coordinate
        circle_points[:, 2] = center[2] + radius * np.sin(angles) + height  # Z coordinate
        circle_points[:, 1] = center[1]  # Y coordinate is constant
    elif plane == "YZ":
        circle_points[:, 1] = center[1] + radius * np.cos(angles)  # Y coordinate
        circle_points[:, 2] = center[2] + radius * np.sin(angles) + height  # Z coordinate
        circle_points[:, 0] = center[0]  # X coordinate is constant
    else:
        raise ValueError("Invalid plane specified. Choose 'XY', 'XZ', or 'YZ'.")

    return circle_points, np.array([[radius, 0, center[2] + radius]]), np.array([-2, -2, 0, 2, 2, 2])


def generate_random_targets(num_targets: int) -> np.ndarray:
    """Generates random targets for the drone to navigate to.

    The targets are generated in a random order and are evenly distributed
    around the origin. The z-coordinate of the targets is randomly chosen
    between 0.1 and 1.0, but is capped at 0.1 if it is below that value.

    Args:
        num_targets: The number of targets to generate.

    Returns:
        A numpy array of shape (num_targets, 3) containing the x, y, and z
        coordinates of the targets.
    """

    targets = np.zeros(shape=(num_targets, 3))
    thetas = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
    phis = np.random.uniform(0.0, 2.0 * math.pi, size=(num_targets,))
    for i, theta, phi in zip(range(num_targets), thetas, phis):
        dist = np.random.uniform(low=1.0, high=1 * 0.9)
        x = dist * math.sin(phi) * math.cos(theta)
        y = dist * math.sin(phi) * math.sin(theta)
        z = abs(dist * math.cos(phi))

        # check for floor of z
        targets[i] = np.array([x, y, z if z > 0.1 else 0.1])

    print(targets)
    return targets


def reaching():
    arr = np.array([
        [-2.5, 4.5, 3],  # G1
        [10, 3.5, 1],  # G2
        [8, -4.5, 1],  # G3
        [-4.5, -6, 2],  # G4
        [-5, -5, 2],  # G5
        [5, -1, 3],  # G6
        [2.5, 6, 3],# G7
        [-2.5, 4.5, 3]
    ])
    # arr = np.array([
    #     [-.75, -0.5, 3.5],  # G1
    #     [8.5, 6, 1],  # G2
    #     [8.5, -3.5, 1],  # G3
    #     [-5, -6, 3.2],  # G4
    #     [-5, -4.5, 1],  # G5
    #     [4, -4.5, 1.3],  # G6
    #     [-2.3, 7, 1.3]])  # G7

    for i, a in enumerate(arr):
        arr[i][2] += 3
        arr[i] /= 5

    return arr, np.array([arr[0]]), np.array([-4, -4, 0, 4, 4, 4])


if __name__ == '__main__':
    targets = up()
    Plotter.plot_3d_targets(targets[0])

    targets = half_up_forward()
    Plotter.plot_3d_targets(targets[0])

    targets = parametric_eq()
    Plotter.plot_3d_targets(targets)

    targets = up_sharp_back_turn()
    Plotter.plot_3d_targets(targets[0])

    targets = up_circle()
    Plotter.plot_3d_targets(targets[0])

    targets = generate_random_targets(10)
    Plotter.plot_3d_targets(targets)

    c = circle(radius=1, num_points=6, height=1, )

    print(c)
    Plotter.plot_3d_targets(c[0])

    point = (0, 0, 2)
    radius = 1
    height = 3
    center = (0, 0, 0)
    plane = "XY"

    # print(is_point_inside_cylinder(point, radius, height, center, plane))

    targets = reaching()
    Plotter.plot_3d_targets(targets[0])
    targets = reaching()
    Plotter.plot_3d_targets(targets[0])

