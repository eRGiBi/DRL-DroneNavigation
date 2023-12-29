from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from gymnasium import spaces
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom network architecture


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, observation_space, net_arch, activation_fn=nn.ReLU):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=net_arch[-1]['shared'])

        # Assuming observation space is a Box space
        self.flatten = nn.Flatten()

        # Fully connected layers for the feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(int(observation_space.shape[0]), net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.shared_layers(x)
        return x


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space,
                 action_space,
                 net_arch,
                 lr_schedule: Callable[[float], float],
                 activation_fn=nn.ReLU,
                 *args,
                 **kwargs):

        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space,
                                                      lr_schedule,
                                                      activation_fn=activation_fn,
                                                      *args, **kwargs)

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False


        # Create the custom feature extractor
        self.features_extractor = CustomFeatureExtractor(observation_space, net_arch, activation_fn)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(net_arch[1], net_arch[2]['pi'][0]),
            activation_fn(),
            nn.Linear(net_arch[2]['pi'][0], net_arch[2]['pi'][1])
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(net_arch[1], net_arch[2]['vf'][0]),
            activation_fn(),
            nn.Linear(net_arch[2]['vf'][0], net_arch[2]['vf'][1])
        )

        self._initialize()

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features).flatten()

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = CategoricalDistribution(action_probs).sample()

        return action, value

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomActorCriticPolicy(self.features_dim)