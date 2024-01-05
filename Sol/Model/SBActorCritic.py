from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
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
    Custom feature extractor for the actor-critic policy.
    It receives as input the features extracted by the features extractor.
    """

    def __init__(self,
                 observation_space,
                 net_arch,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 device: Union[th.device, str] = 'auto',
                 ):
        super(CustomFeatureExtractor, self).__init__(observation_space,
                                                     features_dim=net_arch)

        # Assuming observation space is a Box space
        self.flatten = nn.Flatten()

        # Fully connected layers for the feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(int(observation_space.shape[0]), net_arch[0]),
            activation_fn(),
        )

        for i in range(1, len(net_arch) - 1):
            if isinstance(net_arch[i], int):
                self.shared_layers.add_module(
                    "shared_fc{}".format(i),
                    nn.Linear(net_arch[i], net_arch[i + 1]),
                        )
                self.shared_layers.add_module(
                    "shared_act{}".format(i),
                    activation_fn(),
                    )

    def forward(self, x):
        x = self.flatten(x)
        x = self.shared_layers(x)
        return x


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g., features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args, **kwargs)

        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        # Create the custom feature extractor
        self.features_extractor = CustomFeatureExtractor(observation_space, net_arch, activation_fn)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
        vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        last_shared_layer_dim = net_arch["shared"][-1]

        # Actor head
        self.action_net = nn.Sequential(
            nn.Linear(last_shared_layer_dim, pi_layers_dims[0]),
            activation_fn(),)
        for i in range(1, len(pi_layers_dims) - 2):
            self.action_net.add_module(
                "pi_fc{}".format(i),
                nn.Linear(pi_layers_dims[i], pi_layers_dims[i + 1]),
                    )
            self.action_net.add_module(
                "pi_act{}".format(i),
                activation_fn(),
                )
        self.action_net.add_module(
            "pi_fc_out",
            nn.Linear(pi_layers_dims[-1], self.action_space.shape[0]))  # Last layer matches continuous action space dimensions
        print(self.action_space.shape[0])
        # Critic head
        self.value_net = nn.Sequential(
            nn.Linear(last_shared_layer_dim, vf_layers_dims[0]),
            activation_fn(),
        )
        for i in range(1, len(vf_layers_dims) - 2):
            self.value_net.add_module(
                "vf_fc{}".format(i),
                nn.Linear(vf_layers_dims[i], vf_layers_dims[i + 1]),
                    )
            self.value_net.add_module(
                "vf_act{}".format(i),
                activation_fn(),
                )
        self.value_net.add_module(
            "vf_fc_out",
            nn.Linear(vf_layers_dims[-1], 1))

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
        self.mlp_extractor = CustomFeatureExtractor(
            self.observation_space, self.net_arch, self.activation_fn,
            self.device)
