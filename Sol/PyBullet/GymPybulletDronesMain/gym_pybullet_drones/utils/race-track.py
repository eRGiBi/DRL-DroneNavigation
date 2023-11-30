import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_rectangular_race_track(length, width, height, num_waypoints=50):
    waypoints = []

    # Create the outer rectangle
    for i in range(num_waypoints):
        x = i * length / num_waypoints
        waypoints.append([x, 0, 0])
        waypoints.append([x, width, 0])
        waypoints.append([x, width, height])
        waypoints.append([x, 0, height])

    # Close the loop
    waypoints.append([0, 0, 0])

    return np.array(waypoints)

# Create a rectangular race track
race_track = create_rectangular_race_track(length=1000, width=500, height=100)

# Visualize the race track in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(race_track[:, 0], race_track[:, 1], race_track[:, 2], marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
