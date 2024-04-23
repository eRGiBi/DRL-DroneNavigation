import numpy as np
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from Sol.Model.Environments.DroneEnvironment import DroneEnvironment

from stable_baselines3 import TD3


class AgentController():
    def __init__(self,
                 target_points,
                 gui=False,
                 threshold=1,
                 discount=1,
                 ):

        self._gui = gui
        self._discount = discount
        self._threshold = threshold
        self._target_points = np.array(target_points)

    def train_agent(self):
        """

        """

        # Create a PyBullet environment
        pybullet_env = DroneEnvironment(
            drone=None,
            race_track=None,
            target_points=self._target_points,
            threshold=self._discount,
            discount=self._threshold,
            gui=False)

        train_py_env = DroneEnvironment(drone=None,
                                        race_track=None,
                                        target_points=self._target_points,
                                        threshold=self._discount,
                                        discount=self._threshold,
                                        gui=False
                                        )
        eval_py_env = DroneEnvironment(drone=None,
                                       race_track=None,
                                       target_points=self._target_points,
                                       threshold=self._discount,
                                       discount=self._threshold,
                                       gui=False)

        discrete_action_env = wrappers.ActionDiscretizeWrapper(pybullet_env, num_actions=4)

        # Convert to TF-Agents environment
        tf_env = tf_py_environment.TFPyEnvironment(discrete_action_env)
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


        # Define the underlying Neural Network

        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]

        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        q_net = tf.keras.Sequential(dense_layers + [q_values_layer])

        # Define the Q-network
        # q_net = q_network.QNetwork(
        #     tf_env.observation_spec(),
        #     tf_env.action_spec(),
        #     preprocessing_layers=q_net,
        #     fc_layer_params=fc_layer_params
        # )

        # Define the DQN agent
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )
        agent.initialize()

        return agent

    def define_policies(self, agent):

        eval_policy = agent.policy
        collect_policy = agent.collect_policy

        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())

        # Create a random policy for data collection
        random_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec())

        self.compute_avg_return(eval_env, random_policy, 100)

        # Collect data from the environment
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity
        )
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            random_policy,
            observers=[replay_buffer.add_batch],
            num_steps=collect_steps_per_iteration
        )

    # Initialize the agent and collect initial data
    collect_driver.run()

    # Train the agent
    for _ in range(num_iterations):
        trajectories, _ = collect_driver.run()
        train_loss = agent.train(experience=trajectories)

    def policy_initialisation(self, env):
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1
        )
        model.learn(total_timesteps=1000000)



    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')
        )

    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # See also the metrics module for standard implementations of different metrics.
    # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
