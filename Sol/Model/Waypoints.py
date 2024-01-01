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


def half_up_forward():
    return [
        # np.array([0.0, 0.0, 0.1]),
    #         np.array([0.0, 0.0, 0.2]),
            np.array([0., 0., 0.5]),
            # np.array([0., 0., 0.4]),
            np.array([0., 0., 1]),
            # np.array([0., 0.1, 0.5]),
            np.array([0., 1, 1.5]),
            # np.array([0., 0.3, 0.5]),
            # np.array([0., 0.4, 0.5]),
            # np.array([0., 0.5, 0.5]),
            np.array([0.5, 1.5, 1.5]),
            # np.array([0.2, 0.5, 0.5]),
            # np.array([0.3, 0.5, 0.5]),
            # np.array([0.4, 0.5, 0.5]),
            np.array([1.5, 1.5, 1.5]),
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

def rnd():
    return [
        # np.array([0.5, 0.5, 0.5]),
        # np.array([0.5, 0.0, 0.3]),
        np.array([-1, 0.2, 0.7]),
        np.array([0.3, 0.5, 1.5]),
        np.array([1.5, 1., 1.5]),
        # np.array([1., 0., 1.5]),
        # np.array([.2, 1., .5]),
        # np.array([1.8, 1.5, -1]),
        np.array([1.5, 1., .5]),
        # np.array([1.5, 0.5, 1]),
        # np.array([1, 0.5, 0.5]),
        # np.array([0.5, 0.2, 0.2]),
        # np.array([0.0, 0.0, 0.2]),
    ]


if __name__ == '__main__':
    targets = rnd()
    Plotter.plot_3d_targets(targets)
