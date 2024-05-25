import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.onnx

import torchviz
import graphviz
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



class Plotter:
    pass


