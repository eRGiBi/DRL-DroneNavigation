import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os

import numpy as np
# import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet, tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from Sol.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.enums import Physics

class SoftAcorCritic():

    def __init__(self):
        pass

    def set_up_training(self, env):
        tempdir = tempfile.gettempdir()

        # Use "num_iterations = 1e6" for better results (2 hrs)
        # 1e5 is just so this doesn't take too long (1 hr)
        num_iterations = 100000  # @param {type:"integer"}

        initial_collect_steps = 10000  # @param {type:"integer"}
        collect_steps_per_iteration = 1  # @param {type:"integer"}
        replay_buffer_capacity = 10000  # @param {type:"integer"}

        batch_size = 256  # @param {type:"integer"}

        critic_learning_rate = 3e-4  # @param {type:"number"}
        actor_learning_rate = 3e-4  # @param {type:"number"}
        alpha_learning_rate = 3e-4  # @param {type:"number"}
        target_update_tau = 0.005  # @param {type:"number"}
        target_update_period = 1  # @param {type:"number"}
        gamma = 0.99  # @param {type:"number"}
        reward_scale_factor = 1.0  # @param {type:"number"}

        actor_fc_layer_params = (256, 256)
        critic_joint_fc_layer_params = (256, 256)

        log_interval = 5000  # @param {type:"integer"}

        num_eval_episodes = 20  # @param {type:"integer"}
        eval_interval = 10000  # @param {type:"integer"}

        policy_save_interval = 5000  # @param {type:"integer"}

        # env = tf_py_environment.TFPyEnvironment(env)
        env = suite_pybullet.load(env)
        env.reset()
        PIL.Image.fromarray(env.render())

        print('Observation Spec:')
        print(env.time_step_spec().observation)
        print('Action Spec:')
        print(env.action_spec())


if __name__ == '__main__':
    drone_environment = PBDroneEnv(race_track=None,
                                   target_points=[np.array([0.0, 0.0, 1.0]),
                                                  np.array([8.59735, -3.3286, -6.07256]),
                                                  np.array([1.5974, -5.0786, -4.32256]),
                                                  np.array([3.2474, 3.32137, -2.5725]),
                                                  np.array([1.3474, 1.6714, -2.07256]), ],
                                   threshold=1,
                                   discount=1,
                                   drone=None,
                                   physics=Physics.PYB,
                                   gui=True,
                                   initial_xyzs=np.array([[0, 0, 0]])
                                   )
    sac = SoftAcorCritic()
    sac.set_up_training(drone_environment)
