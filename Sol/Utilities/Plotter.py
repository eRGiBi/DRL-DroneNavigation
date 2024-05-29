import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.onnx

import torchviz
import graphviz
from matplotlib.collections import LineCollection
from torchviz import make_dot

import hiddenlayer as hl


def plot_metrics(episode_rewards, avg_rewards,
                 exploration_rate, episode_durations,
                 losses, title='Learning Metrics'):
    [np.mean(episode_rewards[max(0, i - 10):i + 1]) for i in range(len(episode_rewards))]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(episode_rewards, label='Total Reward', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Reward', color=color)
    ax2.plot(avg_rewards, label='Avg. Reward', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(exploration_rate, label='Exploration Rate', color='green')
    plt.title('Exploration Rate')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(episode_durations, label='Episode Duration', color='orange')
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss', color='purple')
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_learning_curve(scores, title='Rewards'):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Total Reward per Episode')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(loc='upper left')
    plt.show()


def plot_3d_targets(targets):
    """
    Plot the targets in 3D.

    Parameters:
    - targets (list): List of target points, where each target is a NumPy array of [x, y, z] coordinates.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, and Z coordinates from targets
    x, y, z = zip(*targets)

    # Plot the targets
    ax.scatter(x, y, z, c='r', marker='o', label='Targets')

    # Connect the targets with lines
    ax.plot(x, y, z, c='b', linestyle='-', linewidth=1, label='Path')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.show()

def vis_policy(model, env):

    print(model.policy)
    print(model.policy.named_parameters())

    # P
    dummy_input = torch.randn(1, env.observation_space.shape[0])
    # action, _ = model.policy.forward(dummy_input)
    action, _ = model.predict(dummy_input.numpy())
    if isinstance(action, np.ndarray):
        action = torch.tensor(action)
    dot = make_dot(action, params=dict(list(model.policy.named_parameters())))
    dot.render('policy_graph', format='png', outfile="Sol/visual/graphviz_policy_graph.png")

    # onnx
    torch.onnx.export(model.policy, dummy_input.numpy(), "policy_model.onnx")

    # hl
    trans = hl.transforms.Compose([hl.transforms.Prune('Constant')])
    graph = hl.build_graph(model.policy, list(dummy_input.numpy()), transforms=trans)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save('sb3_policy_graph.png', format='png', path="Sol/visual/")


def plot_trajectories(avg_traj, avg_speed):
    fig, ax = plt.subplots()

    traj = np.array(avg_traj)
    speed = np.array(avg_speed)

    # Ensure no negative speeds
    speed = np.maximum(0, speed)

    # Normalize speed for color mapping
    norm = plt.Normalize(speed.min(), speed.max())
    cmap = plt.cm.jet

    points = ax.scatter(traj[:, 0], traj[:, 1], c=speed, cmap=cmap, norm=norm)

    # Add direction arrows
    step = max(1, len(traj) // 5)  # Adjust arrow frequency as needed
    for k in range(1, len(traj), step):
        ax.annotate(
            '',
            xy=(traj[k, 0], traj[k, 1]), xycoords='data',
            xytext=(traj[k-1, 0], traj[k-1, 1]), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5)
        )

    # Add colorbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('Speed (m/s)')

    # Add labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Averaged Drone Trajectory')

    # Add grid and legend
    ax.grid(True)

    # Show the plot
    plt.show()


def plot_trajectories2(avg_traj, avg_speed):
    fig, ax = plt.subplots(figsize=(12, 10))

    traj = np.array(avg_traj)
    speed = np.array(avg_speed)

    # Normalize speed for color mapping
    norm = plt.Normalize(vmin=0, vmax=speed.max())
    cmap = plt.cm.jet

    # Create a continuous line with color gradient
    points = np.array([traj[:, 0], traj[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speed)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    # Add colorbar
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Speed (m/s)')

    # Add direction arrows and labels for key points
    step = max(1, len(traj) // 7)  # Adjust arrow frequency as needed
    for k in range(1, len(traj), step):
        ax.annotate(
            '',
            xy=(traj[k, 0], traj[k, 1]), xycoords='data',
            xytext=(traj[k - 1, 0], traj[k - 1, 1]), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5)
        )
        ax.text(traj[k, 0], traj[k, 1], str(k // step + 1), fontsize=12, color='black')

    # Add labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Averaged Drone Trajectory')

    # Add grid
    ax.grid(True)

    # Set axis limits to better match the reference plot
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)

    # Show the plot
    plt.show()


def plot_3d_trajectories(avg_traj, avg_speed, title="Averaged Drone Trajectory"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    traj = np.array(avg_traj)
    speed = np.array(avg_speed)

    # Normalize speed for color mapping
    norm = plt.Normalize(vmin=0, vmax=speed.max())
    cmap = plt.cm.jet

    # Plot the trajectory with color gradient
    for i in range(len(traj) - 1):
        ax.plot(
            traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
            color=cmap(norm(speed[i])), linewidth=2
        )

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(speed)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
    cbar.set_label('Speed (m/s)')

    # Add direction arrows for key points
    step = max(1, len(traj) // 7)  # Adjust arrow frequency as needed
    for k in range(1, len(traj), step):
        ax.quiver(traj[k-1, 0], traj[k-1, 1], traj[k-1, 2],
                  traj[k, 0] - traj[k-1, 0], traj[k, 1] - traj[k-1, 1], traj[k, 2] - traj[k-1, 2],
                  color='black', lw=1.5)

    # Add labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(title)

    # Add grid
    ax.grid(True)

    # Set axis limits to better match the reference plot
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(0, 0.75)

    # Show the plot
    plt.show()

def plot_all_trajectories_3d(trajectories, speeds, title="All Traveled Paths (3D)"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for traj, speed in zip(trajectories, speeds):
        traj = np.array(traj)
        speed = np.array(speed)

        # Normalize speed for color mapping
        norm = plt.Normalize(vmin=0, vmax=speed.max())
        cmap = plt.cm.jet

        # Plot the trajectory with color gradient
        for i in range(len(traj) - 1):
            ax.plot(
                traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
                color=cmap(norm(speed[i])), linewidth=2
            )

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(speed)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
    cbar.set_label('Speed (m/s)')

    # Add labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(title)

    # Add grid
    ax.grid(True)

    # Set axis limits to better match the reference plot
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(0, 0.75)

    # Show the plot
    plt.show()


class Plotter:
    pass


