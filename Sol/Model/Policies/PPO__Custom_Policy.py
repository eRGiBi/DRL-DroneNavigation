import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)

        # Define custom neural network architecture
        self.custom_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Define actor and critic layers based on the custom network
        self.actor = nn.Linear(64, self.action_space.n)
        self.critic = nn.Linear(64, 1)

    def _features(self, obs):
        # Define the forward pass through the custom neural network
        return self.custom_net(obs)
