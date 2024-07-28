# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# class DroneBoundaryVisualizer:
#     def __init__(self, circle_radius=1, threshold=0.2, target_points=None):
#         self.circle_radius = circle_radius
#         self._threshold = threshold
#         self.circle = True
#         self._current_target_index = 0
#         self.INIT_XYZS = [(0, 0, 0)]
#         self._target_points = target_points or [(1, 1, 1), (2, 2, 2)]
#
#     def current_target(self):
#         return self._target_points[self._current_target_index]
#
#     def is_out_of_cylinder_bounds(self, drone_position, circle_center=(0, 0, 1), extension_length=0.2):
#         drone_vec = np.array(drone_position)
#         center_vec = np.array(circle_center)
#         center_to_drone_vec = drone_vec - center_vec
#         center_to_drone_vec[2] = 0
#
#         try:
#             norm_vec = center_to_drone_vec / np.linalg.norm(center_to_drone_vec) * self.circle_radius
#         except FloatingPointError:
#             norm_vec = np.zeros_like(center_to_drone_vec)
#
#         closest_point = center_vec + norm_vec
#         distance_from_closest_point = np.linalg.norm(drone_position - closest_point)
#
#         if self.circle:
#             return distance_from_closest_point > self._threshold
#         else:
#             if self._current_target_index == 0:
#                 base1 = np.array(self.INIT_XYZS[0])
#                 base2 = np.array(self.current_target())
#             else:
#                 base1 = np.array(self._target_points[self._current_target_index - 1])
#                 base2 = np.array(self.current_target())
#
#             line_vec = base2 - base1
#             line_length = np.linalg.norm(line_vec)
#
#             if line_length == 0:
#                 return np.linalg.norm(drone_position - base1) > self._threshold
#
#             line_unit_vec = line_vec / line_length
#             extended_point1 = base1 - extension_length * line_unit_vec
#             extended_point2 = base2 + extension_length * line_unit_vec
#             point1_to_drone_vec = drone_position - extended_point1
#             projection_length = np.dot(point1_to_drone_vec, line_unit_vec)
#             projection_length = np.clip(projection_length, 0, np.linalg.norm(extended_point2 - extended_point1))
#             closest_point_on_line = extended_point1 + projection_length * line_unit_vec
#             distance_from_line = np.linalg.norm(drone_position - closest_point_on_line)
#             return distance_from_line > self._threshold + extension_length
#
#
# visualizer = DroneBoundaryVisualizer(circle_radius=1, threshold=0.2, target_points=[(1, 1, 1), (2, 2, 2)])
#
# drone_position = (1.5, 0.5, 1.2)
# circle_center = (0, 0, 1)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# theta = np.linspace(0, 2 * np.pi, 100)
# x = circle_center[0] + visualizer.circle_radius * np.cos(theta)
# y = circle_center[1] + visualizer.circle_radius * np.sin(theta)
# z = np.full_like(x, circle_center[2])
# ax.plot(x, y, z, 'b-', label='Circle Boundary')
#
# ax.scatter(drone_position[0], drone_position[1], drone_position[2], color='r', label='Drone Position')
#
# drone_vec = np.array(drone_position)
# center_vec = np.array(circle_center)
# center_to_drone_vec = drone_vec - center_vec
# center_to_drone_vec[2] = 0
# norm_vec = center_to_drone_vec / np.linalg.norm(center_to_drone_vec) * visualizer.circle_radius
# closest_point = center_vec + norm_vec
# ax.scatter(closest_point[0], closest_point[1], closest_point[2], color='g', label='Closest Point on Circle')
#
# base1 = np.array(visualizer.INIT_XYZS[0])
# base2 = np.array(visualizer.current_target())
# line_vec = base2 - base1
# line_unit_vec = line_vec / np.linalg.norm(line_vec)
# extension_length = 0.2
# extended_point1 = base1 - extension_length * line_unit_vec
# extended_point2 = base2 + extension_length * line_unit_vec
# ax.plot([extended_point1[0], extended_point2[0]], [extended_point1[1], extended_point2[1]],
#         [extended_point1[2], extended_point2[2]], 'k--', label='Extended Line Segment')
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
#
# ax.set_box_aspect([1, 1, 1])
#
# plt.show()


import matplotlib.pyplot as plt

def generate_torus(R, r, center_z, num_points=100):

    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)
    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v) + center_z

    return x, y, z

R = 1  # Major
r = 0.5  # Minor
center_z = 1  # Center of the torus along the z-axis
z_axis_length = 3  # Length of the Z axis

x, y, z = generate_torus(R, r, center_z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)

ax.set_zlim(center_z - z_axis_length / 2, center_z + z_axis_length / 2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Boundaries of the Circle Path')

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.linalg import norm
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# path_points = [
#         np.array([0.0, 0.0, 0.5]),
#         np.array([-0.5, 0.2, 0.7]),
#         np.array([0.3, 0.5, 0.7]),
#         np.array([1, 0.5, 1]),
#         np.array([1.5, 1., 1.2])
# ]
#
# R = 0.3  # Minor r
#
# for i in range(len(path_points) - 1):
#     p0 = path_points[i]
#     p1 = path_points[i + 1]
#
#     v = p1 - p0
#     mag = norm(v)
#     v = v / mag
#
#     not_v = np.array([1, 0, 0])
#     if np.all(v == not_v):
#         not_v = np.array([0, 1, 0])
#
#     n1 = np.cross(v, not_v)
#     n1 /= norm(n1)
#     n2 = np.cross(v, n1)
#
#     t = np.linspace(0, mag, 100)
#     theta = np.linspace(0, 2 * np.pi, 100)
#     t, theta = np.meshgrid(t, theta)
#
#     X = p0[0] + v[0] * t + R * np.sin(theta) * n1[0] + R * np.cos(theta) * n2[0]
#     Y = p0[1] + v[1] * t + R * np.sin(theta) * n1[1] + R * np.cos(theta) * n2[1]
#     Z = p0[2] + v[2] * t + R * np.sin(theta) * n1[2] + R * np.cos(theta) * n2[2]
#
#     ax.plot_surface(X, Y, Z, alpha=0.6)
#     ax.plot(*zip(p0, p1), color='red')
#
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(0, 2)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Torus around Arbitrary Path')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def interpolate_path_points(p0, p1, num_points=100):
    """
    Interpolate between two points p0 and p1 along a line segment.
    """
    t = np.linspace(0, 1, num_points)
    interpolated_points = [(1 - ti) * p0 + ti * p1 for ti in t]
    return interpolated_points

def generate_torus(R, r, center_z, num_points=100):
    """
    Generate the coordinates of a torus.
    """
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)
    u, v = np.meshgrid(u, v)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v) + center_z

    return x, y, z

R = 0.5
r = 0.1
center_z = 1

# Define arbitrary points
path_points = [
    np.array([0.0, 0.0, 0.5]),
    np.array([-0.5, 0.2, 0.7]),
    np.array([0.3, 0.5, 0.7]),
    np.array([1, 0.5, 1]),
    np.array([1.5, 1., 1.2])
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(path_points) - 1):
    p0 = path_points[i]
    p1 = path_points[i + 1]

    interpolated_points = interpolate_path_points(p0, p1, num_points=50)

    for point in interpolated_points:
        x, y, z = generate_torus(R, r, point[2], num_points=50)  # Adjust num_points for smoother torus
        ax.plot_surface(x, y, z, color='b', alpha=0.6)

for i in range(len(path_points) - 1):
    p0 = path_points[i]
    p1 = path_points[i + 1]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='red')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Smooth Torus around Arbitrary Path')

plt.show()

