import os
import re
import subprocess
import time
from datetime import datetime

import sys

# from tensorflow.python.types.core import Callable
from typing import Callable

# import pybullet as p
import gym
import ray
import torch.nn
import yaml
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms import PPOConfig


# TODO
sys.path.append("../")
sys.path.append("./")

import numpy as np
import torch as th
import wandb

import stable_baselines3.common.monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import SAC, DDPG
from Sol.Model.Algorithms.sb3_ppo import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement, CheckpointCallback

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from Sol.Model.Environments.PBDroneEnv import PBDroneEnv
from Sol.Model.Environments import normalize
from Sol.Model.Algorithms.utils import *

from Sol.PyBullet.Logger import Logger
import Sol.Utilities.Waypoints as Waypoints
import Sol.Utilities.Callbacks as Callbacks

from Sol.PyBullet.enums import ActionType

from Sol.Utilities.Plotter import plot_learning_curve, plot_trajectories, plot_trajectories2, plot_3d_trajectories, \
    plot_all_trajectories_3d, compute_velocity_acceleration
from Sol.Utilities.Printer import print_ppo_conf, print_sac_conf

# from tf_agents.environments import py_environment
# import tf_agents

# import aim
from wandb.integration.sb3 import WandbCallback


# import rl_zoo3
# import rl_zoo3.train
# from rl_zoo3.train import train
# from sbx import DDPG, PPO, SAC, TQC

# rl_zoo3.ALGOS["sac"] = SAC
# rl_zoo3.ALGOS["ppo"] = PPO
# rl_zoo3.ALGOS["tqc"] = TQC

# import ray
# from ray import tune
# from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.env.env_context import EnvContext


def dilate_targets(targets, factor: int) -> list:
    dilated_points = []

    for i in range(len(targets) - 1):
        start_point = targets[i]
        end_point = targets[i + 1]

        # Generate intermediate points using np.linspace
        intermediate_points = np.linspace(start_point, end_point, num=factor + 2)

        # Append the intermediate points to the dilated points
        dilated_points.extend(intermediate_points[:-1])  # Exclude the last point to avoid duplication

    dilated_points.append(targets[-1])

    return dilated_points


class PBDroneSimulator:
    """
    A class to manage the training and testing of the PBDroneEnv environment.
    """
    def __init__(self, args, track, target_factor=0,
                 plot=True,
                 discount=0.999,
                 ):

        self.args = args
        self.plot = plot
        self.discount = discount
        self.threshold = 0.3
        self.env_steps = args.max_env_steps

        # self.max_reward = 100 + len(targets)

        self.num_envs = args.num_envs

        # Different for each track
        self.track = track
        self.initial_xyzs = track.initial_xyzs
        self.aviary_dim = track.aviary_dim
        self.targets = dilate_targets(track.waypoints, target_factor)
        if track.is_circle:
            self.targets.pop(0)
        print(track)

        self.continued_agent = "Sol/model_chkpts/PPO_save_05.31.2024_03.06.57/best_model.zip"

    def make_env(self, multi=False, gui=False, initial_xyzs=None,
                 aviary_dim=np.array([-1, -1, 0, 1, 1, 1]),
                 rank: int = 0,
                 seed: int = 0,
                 save_path: str = None,
                 include_distance=True,
                 normalize_actions=True,
                 collect_rollouts=False
                 ):
        """
        Utility function for multi-processed env.
        """

        def _init():
            env = PBDroneEnv(
                target_points=self.targets,
                threshold=self.threshold,
                discount=self.discount,
                max_steps=self.env_steps,
                act=ActionType.THRUST,
                # physics=Physics.PYB,
                gui=gui,
                initial_xyzs=initial_xyzs,
                # initial_rpys=np.array([list(self.face_target())]),
                save_folder=save_path,
                aviary_dim=aviary_dim,
                random_spawn=False,
                cylinder=True,
                circle=self.track.is_circle,
                include_distance=include_distance,
                normalize_actions=normalize_actions,
                collect_rollouts=collect_rollouts
            )
            env.reset(seed=seed + rank)

            # env = gym.wrappers.RecordEpisodeStatistics(env)
            # if args.capture_video:
            #     env = gym.wrappers.RecordVideo(env, "results/videos/")

            # env = gym.wrappers.ClipAction(env)
            # env = gym.wrappers.RescaleAction(env, -1, 1)
            env = normalize.NormalizeObservation(env)
            # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

            # TODO: MaxAndSkipEnv

            if self.args.vec_check_nan:
                env = VecCheckNan(env)
            if self.args.vec_normalize:
                env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1)

            if self.args.clip_rew:
                env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            if self.args.norm_rew:
                env = gym.wrappers.NormalizeReward(env)

            env = Monitor(env)

            return env

        if multi:
            set_random_seed(seed)
            return _init
        else:
            return _init()

    def setup_agent(self, tensorboard_path, train_env, chkpt_path=None) -> BaseAlgorithm:
        """
        Initializes the agent with the specified hyperparameters, from scratch or from a saved model.
        """

        if self.args.lib == "sb3":

            if self.args.agent == "RECPPO":
                lstm_kwargs = dict(activation_fn=th.nn.Tanh,
                                   share_features_extractor=True,
                                   net_arch=dict(vf=[512, 512, 256],
                                                 pi=[512, 512, 256]),
                                   enable_critic_lstm=False,
                                   lstm_hidden_size=128,  # 256 default
                                   n_lstm_layers=2,
                                   # log_std_init=0.0,
                                   )

                model = RecurrentPPO(RecurrentActorCriticPolicy,
                                     train_env,
                                     verbose=2,
                                     n_steps=4096,
                                     batch_size=512,
                                     n_epochs=10,
                                     gamma=0.99,
                                     # ent_coef=0.1,
                                     vf_coef=0.5,
                                     gae_lambda=0.9,
                                     # use_sde=True,
                                     # sde_sample_freq=4,
                                     normalize_advantage=True,
                                     clip_range=0.2,
                                     learning_rate=2.5e-4,
                                     tensorboard_log=tensorboard_path if self.args.savemodel else None,
                                     device="auto",
                                     policy_kwargs=lstm_kwargs
                                     )

            if self.args.run_type == "full":

                net_arch = dict(vf=[256, 256], pi=[256, 256]),
                # net_arch = dict(vf=[512, 512, 256, 256], pi=[256, 256])
                net_arch = dict(vf=[512, 512, 256], pi=[512, 512, 256])

                onpolicy_kwargs = dict(activation_fn=th.nn.Tanh,
                                       share_features_extractor=False,
                                       net_arch=net_arch,
                                       # log_std_init=0.5
                                       )

                if self.args.agent == 'PPO':
                    model = PPO(ActorCriticPolicy,
                                env=train_env,
                                verbose=2,
                                n_steps=4096,
                                batch_size=512,
                                n_epochs=10,
                                gamma=0.99,
                                ent_coef=0.02,
                                vf_coef=0.5,
                                clip_range_vf=0.3,
                                gae_lambda=0.95,
                                # use_sde=True,
                                # sde_sample_freq=4,
                                normalize_advantage=True,
                                max_grad_norm=0.5,
                                clip_range=0.2,
                                target_kl=0.05,
                                learning_rate=2.5e-4,
                                # learning_rate=linear_schedule(2.5e-4),
                                # learning_rate=lr_increase(2.5e-4, 5e-4, 0.4),
                                # learning_rate=exponential_schedule(2.5e-4),
                                tensorboard_log=tensorboard_path if self.args.savemodel else None,
                                device="auto",
                                policy_kwargs=onpolicy_kwargs,

                                )
                    print_ppo_conf(model)
                    # print(model.get_parameters())

                elif self.args.agent == 'SAC':

                    offpolicy_kwargs = dict(activation_fn=th.nn.ReLU,
                                            net_arch=[256, 256, ]
                                            )

                    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                            net_arch=dict(qf=[256, 256, 128],
                                                          pi=[256, 256]),
                                            share_features_extractor=False,
                                            )

                    model = SAC(
                        "MlpPolicy",
                        train_env,
                        # replay_buffer_class=HerReplayBuffer,
                        # replay_buffer_kwargs=dict(
                        #     n_sampled_goal=len(self.targets),
                        #     goal_selection_strategy="future",
                        # ),
                        verbose=2,
                        tensorboard_log=tensorboard_path if self.args.savemodel else None,
                        learning_starts=8192,
                        train_freq=3,
                        gradient_steps=5,
                        batch_size=1024,
                        ent_coef="auto",
                        target_update_interval=1,
                        tau=0.005,
                        target_entropy="auto",
                        buffer_size=1_048_576,
                        learning_rate=2.5e-4,
                        gamma=0.99,
                        use_sde=False,
                        # sde_sample_freq=-1,
                        policy_kwargs=offpolicy_kwargs,
                        device="auto",

                    )
                    print_sac_conf(model)

                elif self.args.agent == 'DDPG':
                    model = DDPG("MlpPolicy",
                                 train_env,
                                 verbose=1,
                                 batch_size=1024,
                                 learning_rate=self.args.learning_rate,
                                 train_freq=(10, "step"),
                                 gradient_steps=10,
                                 action_noise=None,
                                 buffer_size=100_000,
                                 learning_starts=1000,
                                 policy_kwargs=dict(net_arch=[256, 256]),

                                 tensorboard_log=tensorboard_path if self.args.savemodel else None,
                                 device="auto",

                                 )

            elif self.args.run_type == "cont":
                print("Train the model from save file. -----------------------------------")

                if self.args.agent == "SAC":
                    model = SAC.load(self.continued_agent, env=train_env)
                    model.load_replay_buffer(os.path.join(load_most_recent_replay_buffer(chkpt_path)))
                    print(model.actor)
                    print(model.critic)
                    print(model.replay_buffer_class)
                    print(model.replay_buffer)
                    print_sac_conf(model)

                elif self.args.agent == "PPO":

                    model = PPO.load(self.continued_agent,
                                     env=train_env,
                                     print_system_info=True)

                    print(model.get_parameters())
                    print_ppo_conf(model)

                elif self.args.agent == "RECPPO":
                    pass
                    # model = RecurrentPPO.load(self.continued_agent,
                    #                           env=train_env,
                    #                           print_system_info=True)
                    # print(model.get_parameters())
                    # print_ppo_conf(model)

        # elif self.args.lib == "ray":
        #
        #     self.ray_train(train_env, )

        return model

    def run_test(self):
        # action = np.array([-.1, -.1, -.1, -.1], dtype=np.float32)
        # action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
        # action = np.array([.9, .9, .9, .9], dtype=np.float32)
        action = np.array([-1, -1, -1, -1], dtype=np.float32)
        action *= -1 / 10
        # action = np.array([0, 0, 0, 0], dtype=np.float32)

        # plot_3d_targets(self.targets)
        self.targets = Waypoints.up()[0]

        drone_environment = self.make_env(gui=True,
                                          initial_xyzs=np.array([[0, 0, 0.1]]),
                                          aviary_dim=np.array([-2, -2, 0, 2, 2, 2]))

        print(drone_environment.G)
        print(drone_environment.INIT_XYZS)

        check_env(drone_environment, warn=True)

        time_step = drone_environment.reset()

        print('[INFO] Action space:', drone_environment.action_space)
        print('[INFO] Observation space:', drone_environment.observation_space)
        print(spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32))

        rewards = []
        rewards_sum = []

        print(time_step)
        i = 0

        while True:
            i += 1
            print("step: ", i, "------------------")
            time_step = drone_environment.step(action)
            rewards.append(time_step[1])
            rewards_sum.append(sum(rewards))
            print(time_step)
            if time_step[2]:
                break

            time.sleep(1. / 240.)  # Control the simulation speed

        # plot_learning_curve(rewards)
        # plot_learning_curve(rewards_sum)

    def test_saved(self):
        """
        Tests the best model saved during a training process.
        Might need to tune the observation and action spaces for each model to work.
        """
        # drone_environment = self.make_env(gui=True,
        #                                   aviary_dim=self.aviary_dim,
        #                                   initial_xyzs=self.initial_xyzs,
        #                                   include_distance=True,
        #                                   normalize_actions=True
        #                                   )
        from Sol.Model.Environments.dum import PBDroneEnv
        drone_environment = PBDroneEnv(gui=False,
                                       target_points=self.targets,
                                       threshold=self.threshold,
                                       discount=self.discount,
                                       max_steps=self.env_steps,
                                       aviary_dim=np.array([-2, -2, 0, 2, 2, 2]),
                                       initial_xyzs=self.initial_xyzs)
        print(drone_environment.observation_space)
        print(drone_environment.action_space)

        saved_filename = "Sol/model_chkpts/save-05.12.2024_17.15.50/best_model.zip"
        saved_filename = "Sol/model_chkpts/PPO_save_05.15.2024_17.34.06/best_model.zip"
        # saved_filename = "Sol/model_chkpts/PPO_save_05.15.2024_00.03.17/best_model.zip"
        # saved_filename = "Sol/model_chkpts/save-05.11.2024_11.37.31/best_model.zip"
        # saved_filename = "Sol/model_chkpts/PPO_save_05.16.2024_09.37.34/best_model.zip"
        # saved_filename = "Sol/model_chkpts/PPO_save_05.19.2024_15.36.44/best_model.zip"  #very good
        # saved_filename = "Sol/model_chkpts/PPO_save_05.19.2024_23.11.04/best_model.zip"  # decent
        # saved_filename = "Sol/model_chkpts/SAC_save_05.26.2024_00.12.30/best_model.zip"
        # saved_filename = "Sol/model_chkpts/SAC_save_05.21.2024_23.28.56/best_model.zip"
        # wrong observation spaces,

        # saved_filename = "Sol/model_chkpts/PPO_save_05.28.2024_18.50.29/best_model.zip"
        saved_filename = "Sol/model_chkpts/exp/ver.zip"

        # The learning processes of these models are good,
        # yet something is wrong with checkpointing them or loading them.
        # saved_filename = "Sol/model_chkpts/PPO_save_05.30.2024_23.25.58/best_model.zip"
        # saved_filename = "Sol/model_chkpts/PPO_save_05.31.2024_03.06.57/best_model.zip"
        # saved_filename = "Sol/model_chkpts/PPO_save_05.31.2024_04.57.44/rl_model_2652000_steps.zip"

        if self.args.agent == "SAC":
            model = SAC.load(saved_filename, env=drone_environment)
            # print(model.get_parameters())
            print(model.actor)
            print(model.critic)
            print(model.replay_buffer_class)
            print_sac_conf(model)

        elif self.args.agent == "PPO":
            model = PPO.load(saved_filename, env=drone_environment, tb_log_name="test")
            model.learn(100_000)
            # print(model.get_parameters())
            print_ppo_conf(model)
            # vis_policy(model, drone_environment)

        # model = PPO.load(os.curdir + "\model_chkpts\success_model.zip")
        # model = SAC.load(os.curdir + "\model_chkpts\success_model.zip")

        obs, info = drone_environment.reset()

        # drone_environment = gym.wrappers.RecordEpisodeStatistics(drone_environment)

        for target in self.targets:
            print(target)

        # self.origin_log(model, saved_filename, drone_environment)

        # Print training progression ############################
        #     with np.load(filename+'/evaluations.npz') as data:
        #         for j in range(data['timesteps'].shape[0]):
        #             print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

        for b in [False, True]:
            times = []
            all_trajectories = []
            all_speeds = []

            rewards = []
            rewards_sum = []

            for j in range(50):
                i = 0
                terminated = False
                drone_environment.reset()

                traj = []
                speed = []

                start = time.time()
                while not terminated:
                    action, _states = model.predict(obs, deterministic=b)

                    obs, reward, terminated, truncated, info = drone_environment.step(action)

                    print("Step:", i, "of deterministic:", b, "------------------", j)
                    i += 1
                    print("Obs:", obs, "\nAction", action, "\nReward:", reward, "\nTerminated:", terminated,
                          "\nTruncated:", truncated, "\nInfo:", info)
                    print("rpy", drone_environment.rpy)
                    print("pos", drone_environment.pos[0])

                    # traj.append(obs[:3])
                    # speed.append(np.linalg.norm(drone_environment.current_vel))
                    # print("Speed:", np.linalg.norm(drone_environment.current_vel))

                    rewards.append(reward)
                    rewards_sum.append(sum(rewards))

                    if terminated:
                        end = time.time()
                        all_trajectories.append(traj)
                        all_speeds.append(speed)

                        print(end - start)
                        times.append(end - start)
                        # plot_learning_curve(rewards)
                        # plot_learning_curve(rewards_sum, title="Cumulative Rewards")
                        print("Cumulative Rewards", sum(rewards))

                        drone_environment.reset()
                        rewards.clear()
                        rewards_sum.clear()

                        # time.sleep(1. / 240.)
                        break
            print(times)
            print(np.average(times))

            aver, s = pad_and_average_trajectories(all_trajectories, all_speeds)
            plot_trajectories(aver, s)
            plot_trajectories2(aver, s)

            plot_3d_trajectories(aver, s)
            plot_all_trajectories_3d(all_trajectories, all_speeds)

            compute_velocity_acceleration(all_trajectories, all_speeds)

            break
            # time.sleep(1. / 240.)

    def test_learning(self):

        """
            For custom policy testing purposes.
        """

        train_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i,
                                                 aviary_dim=np.array([-2, -2, 0, 2, 2, 2]))
                                   for i in range(1)]
                                  )

        # custom_policy = CustomActorCriticPolicy(train_env.observation_space, train_env.action_space,
        #                                         net_arch=[512, 512, dict(vf=[256, 128],
        #                                                                  pi=[256, 128])],
        #                                         lr_schedule=linear_schedule(1e-3),
        #                                         activation_fn=th.nn.Tanh)

        onpolicy_kwargs = dict(net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])])

        model = PPO("MlpPolicy",
                    train_env,
                    verbose=1,
                    n_steps=self.args.max_env_steps,
                    batch_size=self.args.batch_size,
                    device="auto",
                    policy_kwargs=onpolicy_kwargs
                    # dict(net_arch=[256, 256, 256], activation_fn=th.nn.GELU, ),
                    )
        print(model.policy)

        model.learn(total_timesteps=int(5e2), )

        # model = PPO(custom_policy,
        #             train_env,
        #             verbose=1,
        #             n_steps=2048,
        #             batch_size=49
        #             # dict(net_arch=[256, 256, 256], activation_fn=th.nn.GELU, ),
        #             )

    def run_full_training(self):
        """ Run the full training process."""

        # # tensorboard --logdir ./Sol/logs/

        if self.args.lib == "clrl" or self.args.lib == "ray":
            print("Running with", self.args.lib)
            print("Registering gym environment")
            import gymnasium as gym

            gym.register(
                id='PBDroneEnv-v0',
                entry_point='Sol.Model.Environments.PBDroneEnv:PBDroneEnv',  # Ensure this matches the module and class
                # max_episode_steps=1000,
                kwargs={
                    'multi': True,
                    'gui': False,
                    'rank': 0,
                    'aviary_dim': self.aviary_dim,
                    'initial_xyzs': self.initial_xyzs
                }
            )
            print(gym.envs.registry)

            if self.args.lib == "clrl":
                self.run_clrl()
                exit()
            elif self.args.lib == "ray":
                self.ray_train()
                exit()

        now = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

        if self.args.savemodel:
            chckpt_path, tensorboard_path, wandb_path = self.setup_paths(now)
            print(chckpt_path, tensorboard_path, wandb_path)
        else:
            chckpt_path, tensorboard_path, wandb_path = None, None, None

        train_env = SubprocVecEnv([self.make_env(multi=True,
                                                 gui=False,
                                                 rank=i,
                                                 aviary_dim=self.aviary_dim,
                                                 initial_xyzs=self.initial_xyzs,
                                                 include_distance=True,
                                                 normalize_actions=True
                                                 )
                                   for i in range(self.num_envs)
                                   ])

        eval_env = SubprocVecEnv([self.make_env(multi=True,
                                                save_path=chckpt_path if self.args.savemodel else None,
                                                gui=self.args.gui,
                                                initial_xyzs=self.initial_xyzs,
                                                aviary_dim=self.aviary_dim,
                                                include_distance=True,
                                                normalize_actions=True,
                                                collect_rollouts=False
                                                )
                                  ])

        print(eval_env.observation_space)
        print(eval_env.action_space)
        # check_env(train_env, warn=True, skip_render_check=True)

        if self.args.lib == "sb3":
            model = self.setup_agent(tensorboard_path, train_env, chkpt_path=chckpt_path)

        model.set_random_seed(42)

        # if self.args.save_model:
        #     with open(os.path.join(chckpt_path, 'config.yaml'), 'w', encoding='UTF-8') as file:
        #         yaml.dump(munch.unmunchify(self.args), file, default_flow_style=False)

        callbacks = []

        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100_000, verbose=1)
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

        if self.args.savemodel:
            # callbacks.append(Callbacks.FoundTargetsCallback(log_dir=chckpt_path + '/'))
            # callbacks.append(CheckpointCallback(save_freq=1000, save_path=chckpt_path + '/',
            #                  save_replay_buffer=True if (self.args.agent == "SAC") else False, verbose=2))

            if self.args.agent == 'SAC':
                callbacks.append(Callbacks.SaveReplayBufferCallback(save_freq=100_000,
                                                                    save_path=chckpt_path,
                                                                    verbose=1)
                                 )

        if self.args.wandb:
            callbacks.append(
                WandbCallback(gradient_save_freq=100, model_save_path=wandb_path, verbose=2, ))

        # AimCallback(repo='.Aim/', experiment_name='sb3_test')

        callbacks.append(
            EvalCallback(eval_env,
                         # callback_on_new_best=callback_on_best,
                         best_model_save_path=chckpt_path if self.args.savemodel else None,
                         log_path=chckpt_path if self.args.savemodel else None,
                         eval_freq=max(1000 // self.num_envs, 1),
                         n_eval_episodes=10,
                         deterministic=False,
                         render=False,
                         verbose=1,
                         )
        )

        # # set up logger
        #         # new_logger = configure(tensorboard_path, ["stdout", "csv", "tensorboard"])
        #         # model.set_logger(new_logger)

        trained_model = model.learn(
            total_timesteps=self.args.total_timesteps,
            callback=callbacks,
            log_interval=1000,
            # tb_log_name=self.args.agent + "_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        )

        if self.args.savemodel:
            model.save(chckpt_path + '/success_model.zip')

            stats_path = os.path.join(chckpt_path, "vec_normalize.pkl")
            eval_env.save(stats_path)

            if self.args.wandb:
                wandb.finish()

        if True:
            self.origin_log(trained_model, chckpt_path, train_env)

    def origin_log(self, model, chckpt_path, train_env):

        test_env = self.make_env(multi=False)
        test_env_nogui = self.make_env(multi=False)

        test_env = Monitor(test_env)
        test_env_nogui = Monitor(test_env_nogui)

        rewards = []

        # if os.path.isfile(filename + '/success_model.zip'):
        #     path = filename + '/success_model.zip'
        # elif os.path.isfile(filename + '/best_model.zip'):
        #     path = filename + '/best_model.zip'
        # else:
        #     print("[ERROR]: no model under the specified path", filename)
        # model = PPO.load(path)

        train_env.close()

        logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                        num_drones=1,
                        # output_folder=(chckpt_path + "/logs/logger/") if self.args.savemodel else None,
                        )

        mean_reward, std_reward = evaluate_policy(model,
                                                  test_env_nogui,
                                                  n_eval_episodes=100
                                                  )
        print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

        obs, info = test_env.reset()
        i = 0
        while True:
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )

            obs, reward, terminated, truncated, info = test_env.step(action)
            rewards.append(reward)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
                  truncated)

            logger.log(drone=0,
                       timestamp=i / test_env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                       control=np.zeros(12))

            if terminated:
                plot_learning_curve(rewards)
                break
            i += 1

        test_env.close()

        if self.plot:
            logger.plot()

    def ray_train(self):
        """"""

        ray.init(ignore_reinit_error=True)

        env_config = {
            "target_points": self.targets,  # Assuming this is defined in your context
            "threshold": self.threshold,
            "discount": self.discount,
            "max_steps": self.args.max_env_steps,
            "aviary_dim": [-2, -2, 0, 2, 2, 2],
            "initial_xyzs": self.initial_xyzs,
            "multi": True,
            "gui": False,
        }

        config = {
            "env": PBDroneEnv,
            "env_config": env_config,
            "num_workers": 1,
            "framework": "torch",
            "train_batch_size": 512,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "gamma": 0.99,
            "vf_loss_coeff": 0.5,
            "lambda": 0.9,
            "clip_param": 0.2,
            "lr": 2.5e-4,
        }

        config = (
            PPOConfig()
            .environment(env="PBDroneEnv-v0", env_config=env_config)
            .env_runners(num_env_runners=2)

            .training(
                model={"fcnet_hiddens": [64, 64]},
                gamma=0.99,
                lambda_=0.95,
                use_critic=True,
                lr=2.5e-4,
                num_sgd_iter=10,
                vf_loss_coeff=0.01,
                use_kl_loss=True,
                clip_param=0.2,
                use_critics=True,
                grad_clip=0.5,
                vf_clip_param=0.3,
                entropy_coeff=0.01,
                shuffle_sequences=True,
                mini_batch_size_per_learner=512,
                kl_target=0.005,

            )
            .evaluation(
                evaluation_num_env_runners=1,
                evaluation_interval=1,
                evaluation_parallel_to_training=True,
            )
        )

        stop = {
            "timesteps_total": self.args.total_timesteps,
        }

        from ray.rllib.algorithms.ppo import PPO

        algo = config.build()

        for _ in range(5):
            print(algo.train())

        algo.evaluate()

        analysis = tune.run(
            PPO,
            config=config.to_dict(),
            stop=stop,
            verbose=1,
        )

        best_checkpoint = analysis.get_best_checkpoint(analysis.get_best_trial(), metric="episode_reward_mean")

        # Load the best checkpoint
        trainer = PPO(config=config)
        trainer.restore(best_checkpoint)

    def run_clrl(self):
        if self.args.agent == "PPO":
            clean_args = [
                "python", "Sol/Model/Algorithms/cleanRLPPO.py",
                "--env-id", "PBDroneEnv-v0",
                "--num-minibatches", "1",
                "--minibatch-size", "512",
                "--update-epochs", "10",
                "--learning-rate", "2.5e-4",
                "--no-anneal-lr",
                "--gamma", "0.99",
                "--gae-lambda", "0.95",
                "--norm-adv",
                "--vf-coef", "0.5",
                "--ent-coef", "0.01",
                "--clip-coef", "0.2",
                "--max-grad-norm", "0.5",
                "--target-kl", "0.05",
                "--num-steps", str(self.args.max_env_steps),
                "--total-timesteps", str(int(self.args.total_timesteps)),
                "--no-save-model",
                # "--no-wandb",
                "--num-envs", str(self.num_envs),
            ]

        subprocess.run(clean_args)

    def face_target(self):
        target_vector = np.array(self.targets[0]) - np.array(self.initial_xyzs[0])
        target_vector_xy = np.array([target_vector[0], target_vector[1]])  # Projection on the XY plane
        yaw = np.arctan2(target_vector[1], target_vector[0])  # Angle in the XY plane

        # Calculate the distance and pitch
        distance = np.linalg.norm(target_vector)
        print(target_vector, distance)
        pitch = np.arcsin(target_vector[2] / distance)  # Angle with respect to the horizontal plane

        return 0, pitch, yaw

    def setup_paths(self, now, base_dir='Sol'):

        base_chkpt_path = os.path.join(base_dir, 'model_chkpts')
        base_log_path = os.path.join(base_dir, 'logs')

        if self.args.run_type == "cont":
            chckpt_path = self.continued_agent.rstrip("best_model.zip")
            tensorboard_path = os.path.join(base_log_path,
                                            self.continued_agent.lstrip(str(base_chkpt_path)).lstrip("PPO_save_"))
        else:
            unique_path = self.args.agent + '_save_' + now
            chckpt_path = os.path.join(base_chkpt_path, unique_path)
            tensorboard_path = os.path.join(base_log_path, unique_path)

        os.makedirs(chckpt_path, exist_ok=True)
        os.makedirs(tensorboard_path, exist_ok=True)

        wandb_path = ''
        if self.args.wandb:
            wandb_path = os.path.join("Sol/wandb" + self.args.agent)
            os.makedirs(wandb_path, exist_ok=True)

        return chckpt_path, tensorboard_path, wandb_path

        # Counting

        # model_count = sum(len(files) for _, _, files in os.walk('Sol/model_chkpts/'))
        # model_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in os.listdir(directory_path))
        # log_count = sum(len(files) for _, _, files in os.walk('Sol/log/'))

        # if self.args.run_type == "cont":
        #     chckpt_path = os.path.join("./Sol/model_chkpts", self.args.agent + '_save_' + str(model_count - 1))
        #     tensorboard_path = os.path.join("./Sol/logs/", self.args.agent + " " + str(log_count - 1))
        # else:
        #     chckpt_path = os.path.join("./Sol/model_chkpts", self.args.agent + '_save_' + str(model_count))
        #     tensorboard_path = os.path.join("./Sol/logs/", self.args.agent + " " + str(log_count - 1))

        # if self.args.wandb:
        #     wandb_path = tensorboard_path + "/wand/"
        #     if not os.path.exists(wandb_path):
        #         os.makedirs(wandb_path + '/')
        # else:
        #     wandb_path = ""
        #
        # if not os.path.exists(chckpt_path):
        #     os.makedirs(chckpt_path + '/')
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path + '/')
        #
        # return chckpt_path, tensorboard_path, wandb_path


def load_most_recent_replay_buffer(directory) -> str:
    """
    Load the recentest replay buffer from a given directory.

    Parameters:
    model (BaseAlgorithm): The RL model for which the replay buffer should be loaded.
    directory (str | Path): Path to the directory containing the replay buffer files.

    Returns:
    Path
    """
    replay_buffer_files = os.listdir(directory)
    replay_buffer_pattern = re.compile(r'replay_buffer_(\d+)\.pkl')

    highest_count = -1
    highest_file = None

    for file in replay_buffer_files:
        match = replay_buffer_pattern.match(file)
        if match:
            count = int(match.group(1))
            if count > highest_count:
                highest_count = count
                highest_file = file

    if not highest_file:
        print("No replay buffer files found.")

    return os.path.join(directory, highest_file.strip('.pkl'))


def pad_and_average_trajectories(trajectories, speeds):
    max_length = max(len(traj) for traj in trajectories)

    padded_trajectories = []
    padded_speeds = []

    for traj, speed in zip(trajectories, speeds):
        padded_traj = traj + [traj[-1]] * (max_length - len(traj))
        padded_speed = speed + [speed[-1]] * (max_length - len(speed))
        padded_trajectories.append(padded_traj)
        padded_speeds.append(padded_speed)

    padded_trajectories = np.array(padded_trajectories)
    padded_speeds = np.array(padded_speeds)

    avg_traj = np.mean(padded_trajectories, axis=0)
    avg_speed = np.mean(padded_speeds, axis=0)

    # Ensure floating-point precision
    avg_speed = np.maximum(0, avg_speed)

    return avg_traj, avg_speed
