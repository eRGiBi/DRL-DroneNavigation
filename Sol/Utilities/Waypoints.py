import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Sol.Utilities.Plotter as Plotter


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
    ]


def half_up_forward():
    return [
        np.array([0., 0., 0.5]),
        np.array([0., 0., 1]),
        np.array([0., 1, 1.5]),
        # np.array([0.5, 1.5, 1.5]),
        # np.array([1.5, 1.5, 1.5]),
    ]


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
    ]


def up_sharp_back_turn():
    return [
        np.array([0.0, 0.0, 0.5]),
        np.array([-0.5, 0.2, 0.7]),
        np.array([0.3, 0.5, 0.7]),
        np.array([1, 0.5, 1]),
        np.array([1.5, 1., 1.2])
    ]


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


if __name__ == '__main__':
    targets = up()
    Plotter.plot_3d_targets(targets)

    targets = half_up_forward()
    Plotter.plot_3d_targets(targets)

    targets = parametric_eq()
    Plotter.plot_3d_targets(targets)

    targets = up_sharp_back_turn()
    Plotter.plot_3d_targets(targets)

    targets = up_circle()
    Plotter.plot_3d_targets(targets)

    targets = generate_random_targets(10)
    Plotter.plot_3d_targets(targets)

