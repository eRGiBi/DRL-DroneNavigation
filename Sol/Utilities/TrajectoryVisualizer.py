
"""
Trajectory visualization.

Dependent on the reward function.

Requires collected data from the environment, stored in a text file.
I used a 2.6 GB file from the PBDroneEnv environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


sample_size = 50
filename = 'Sol/rollouts/rollout_9.txt'


# with open(filename, 'r') as file:
#     data = [list(map(float, line.split(','))) for line in file]
#
# sequences = []
# current_sequence = []
# for row in data:
#     current_sequence.append(row)
#     if row[-1] == -10 or row[-1] == 8:
#         sequences.append(current_sequence)
#         current_sequence = []
#
# sampled_sequences = sequences[::max(1, len(sequences) // sample_size)][:sample_size]
#
# trajectories = []
# for sequence in sampled_sequences:
#     trajectories.append([step[:3] for step in sequence])
#
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# for traj in trajectories:
#     traj = np.array(traj)
#     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
#
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
# ax.set_title('3D Trajectories')
#
# ax.grid(True)
#
# ax.set_xlim(-0.75, 0.75)
# ax.set_ylim(-0.75, 0.75)
# ax.set_zlim(0, 0.75)
#
# plt.show()



sample_size = 100
with open(filename, 'r') as file:
    data = [list(map(float, line.split(','))) for line in file]

sequences = []
current_sequence = []
for row in data:
    current_sequence.append(row)
    if row[-1] == -10 or row[-1] == 8:
        sequences.append(current_sequence)
        current_sequence = []

sampled_sequences = sequences[::max(1, len(sequences) // sample_size)][:sample_size]

trajectories = []
for sequence in sampled_sequences:
    trajectories.append([step[:3] for step in sequence])

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

cmap = plt.cm.jet
colors = cmap(np.linspace(0, 1, len(trajectories)))  # Generate unique colors for each trajectory

for traj, color in zip(trajectories, colors):
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(trajectories)))
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
cbar.set_label('Sequence Order')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('3D Trajectories with Sequence Order Color')

ax.grid(True)

ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.75, 0.75)
ax.set_zlim(0, 0.75)

plt.show()



sequences = []
current_sequence = []
for row in data:
    current_sequence.append(row)
    if row[-1] == -10 or row[-1] == 8:
        sequences.append(current_sequence)
        current_sequence = []

sample_size = 5000
sampled_sequences = sequences[::max(1, len(sequences) // sample_size)][:sample_size]

trajectories = []
for sequence in sampled_sequences:
    trajectories.append([step[:3] for step in sequence])

fig, ax = plt.subplots(figsize=(12, 10))

cmap = plt.cm.jet
colors = cmap(np.linspace(0, 1, len(trajectories)))  # Generate unique colors for each trajectory

for traj, color in zip(trajectories, colors):
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(trajectories)))
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
cbar.set_label('Sequence Order')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('2D Trajectories with Sequence Order Color')

ax.grid(True)

ax.set_xlim(-0.75, 1.2)
ax.set_ylim(-0.5, )

plt.show()
