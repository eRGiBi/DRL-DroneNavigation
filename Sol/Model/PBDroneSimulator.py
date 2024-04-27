import os
import time
from datetime import datetime

import sys

# from tensorflow.python.types.core import Callable
from typing import Callable

import gym
import torch.distributions

# TODO
sys.path.append("../")
sys.path.append("./")

import numpy as np
import torch as th
import wandb

from stable_baselines3.common.env_checker import check_env
import stable_baselines3.common.monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed

from Sol.PyBullet.enums import Physics
# from Sol.DroneEnvironment import DroneEnvironment
from Sol.Model.Environments.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.Logger import Logger
import Sol.Model.Waypoints as Waypoints

from Sol.Utilities.Plotter import plot_learning_curve, plot_3d_targets
import Sol.Utilities.Callbacks as Callbacks
from Sol.Utilities.Printer import print_ppo_conf, print_sac_conf

# from tf_agents.environments import py_environment

# import aim
from wandb.integration.sb3 import WandbCallback


def dilate_targets(targets, factor: int) -> list:
    # Initialize an empty array to store the dilated points
    dilated_points = []

    # Iterate over each pair of consecutive points
    for i in range(len(targets) - 1):
        start_point = targets[i]
        end_point = targets[i + 1]

        # Generate intermediate points using np.linspace
        intermediate_points = np.linspace(start_point, end_point, num=factor + 2)

        # Append the intermediate points to the dilated points
        dilated_points.extend(intermediate_points[:-1])  # Exclude the last point to avoid duplication

    # Add the last target point to the result
    dilated_points.append(targets[-1])

    return dilated_points


class PBDroneSimulator:

    def __init__(self, args, targets, target_factor=0,
                 plot=True,
                 discount=0.999,
                 ):

        self.args = args
        self.plot = plot
        self.discount = discount
        self.threshold = args.threshold
        self.env_steps = args.max_env_steps

        # self.max_reward = 100 + len(targets) * 10

        self.num_envs = args.num_envs
        self.targets = dilate_targets(targets, target_factor)

        self.initial_xyzs = np.array([[0, 0, 0.2]])

    def make_env(self, multi=False, gui=False, initial_xyzs=None,
                 aviary_dim=np.array([-1, -1, 0, 1, 1, 1]),
                 rank: int = 0, seed: int = 0,
                 save_path: str = None):
        """
        Utility function for multi-processed env.
        """

        def _init():
            env = PBDroneEnv(
                target_points=self.targets,
                threshold=self.threshold,
                discount=self.discount,
                max_steps=self.env_steps,
                physics=Physics.PYB,
                gui=gui,
                initial_xyzs=initial_xyzs,
                save_folder=save_path,
                aviary_dim=aviary_dim,
                random_spawn=True,
            )
            env.reset(seed=seed + rank)
            env = Monitor(env)  # record stats such as returns
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            # if args.capture_video:
            #     env = gym.wrappers.RecordVideo(env, "results/videos/")
            # env = gym.wrappers.ClipAction(env)
            # env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            # env = gym.wrappers.NormalizeReward(env)
            # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env

        if multi:
            set_random_seed(seed)
            return _init
        else:
            return _init()

    def run_test(self):
        # action = np.array([-.1, -.1, -.1, -.1], dtype=np.float32)
        # action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
        # action = np.array([.9, .9, .9, .9], dtype=np.float32)
        action = np.array([-1, -1, -1, -1], dtype=np.float32)
        action *= -1 / 10
        # action = np.array([0, 0, 0, 0], dtype=np.float32)

        plot_3d_targets(self.targets)
        self.targets = Waypoints.up()

        drone_environment = self.make_env(gui=True,  # initial_xyzs=np.array([[0, 0, 0.5]]),
                                          aviary_dim=np.array([-2, -2, 0, 2, 2, 2]))
        print(drone_environment.G)

        print(drone_environment.INIT_XYZS)

        # It will check your custom environment and output additional warnings if needed
        # check_env(drone_environment, warn=True)

        time_step = drone_environment.reset()

        print('[INFO] Action space:', drone_environment.action_space)
        print('[INFO] Observation space:', drone_environment.observation_space)

        rewards = []
        rewards_sum = []

        print(time_step)
        i = 0

        # for _ in range(100):
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

        plot_learning_curve(rewards)
        plot_learning_curve(rewards_sum)

    def test_saved(self):
        drone_environment = self.make_env(gui=True, aviary_dim=np.array([-2, -2, 0, 2, 2, 2]))

        saved_filename = "Sol/model_chkpts/save-04.24.2024_16.22.21/best_model.zip"

        if self.args.agent == "SAC":
            model = SAC.load(saved_filename)
            # print(model.get_parameters())
            print(model.actor)
            print(model.critic)
            print(model.replay_buffer_class)
            print_sac_conf(model)

        if self.args.agent == "PPO":
            model = PPO.load(saved_filename)
            # print(model.get_parameters())
            print_ppo_conf(model)

        # model = PPO.load(os.curdir + "\model_chkpts\success_model.zip")
        # model = SAC.load(os.curdir + "\model_chkpts\success_model.zip")

        rewards = []
        rewards_sum = []
        obs, info = drone_environment.reset()

        # drone_environment = gym.wrappers.RecordEpisodeStatistics(drone_environment)

        for target in self.targets:
            print(target)

        #### Print training progression ############################
        #     with np.load(filename+'/evaluations.npz') as data:
        #         for j in range(data['timesteps'].shape[0]):
        #             print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

        for _ in range(5):
            for b in [True, False]:

                for i in range(self.max_steps):
                    action, _states = model.predict(obs,
                                                    deterministic=b
                                                    )
                    obs, reward, terminated, truncated, info = drone_environment.step(action)
                    print(i)
                    print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated,
                          "\tTruncated:",
                          truncated)
                    print("rpy", drone_environment.rpy)
                    print("pos", drone_environment.pos[0])
                    rewards.append(reward)
                    rewards_sum.append(sum(rewards))

                    if terminated:
                        plot_learning_curve(rewards)
                        plot_learning_curve(rewards_sum, title="Cumulative Rewards")
                        print("Cumulative Rewards", sum(rewards))

                        drone_environment.reset()
                        rewards.clear()
                        rewards_sum.clear()
                        break

                    # time.sleep(1. / 240.)

    def test_learning(self):

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
        start = time.perf_counter()

        tensorboard_path = "./Sol/logs/" + self.args.agent + " " + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

        chckpt_path = os.path.join("./Sol/model_chkpts", 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))

        if not os.path.exists(chckpt_path):
            os.makedirs(chckpt_path + '/')

        train_env = self.make_env(multi=False, gui=False)
        #    check_env(train_env, warn=True, skip_render_check=True)

        train_env = SubprocVecEnv([self.make_env(multi=True,
                                                 gui=False,
                                                 rank=i,
                                                 aviary_dim=np.array([-2, -2, 0, 2, 2, 2]),
                                                 initial_xyzs=self.initial_xyzs,
                                                 )
                                   for i in range(self.args.num_envs)]
                                  )

        # eval_env = make_env(multi=False, gui=False, rank=0)

        eval_env = SubprocVecEnv([self.make_env(multi=True,
                                                save_path=chckpt_path if self.args.savemodel else None,
                                                gui=self.args.gui,
                                                aviary_dim=np.array([-2, -2, 0, 2, 2, 2])), ])
        # eval_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i) for i in range(self.num_cpu)])

        if self.args.vec_check_nan:
            train_env = VecCheckNan(train_env)
            eval_env = VecCheckNan(eval_env)

        if self.args.vec_normalize:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                     clip_obs=1)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
                                    clip_obs=1)

        # On-policy algorithms #################################

        onpolicy_kwargs = dict(activation_fn=th.nn.Tanh,
                               share_features_extractor=True,
                               net_arch=dict(vf=[256, 256],
                                             pi=[256, 256]),
                               )

        custom_policy = dict(net_arch=[dict(share=[512, 512], vf=[256, 128], pi=[256, 128])],
                             activation_fn=th.nn.Tanh)

        # Off-policy algorithms #################################
        offpolicy_kwargs = dict(activation_fn=th.nn.ReLU,
                                net_arch=[512, 512, 256, 128])

        #     offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                             dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))

        if self.args.agent == 'PPO':
            model = PPO(ActorCriticPolicy,
                        env=train_env,
                        verbose=1,
                        n_steps=4096,
                        batch_size=self.args.batch_size,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        gae_lambda=0.9,
                        # use_sde=True,
                        # sde_sample_freq=4,
                        normalize_advantage=True,
                        clip_range=0.1,
                        learning_rate=int(self.args.learning_rate),
                        tensorboard_log=(tensorboard_path + "/ppo_tensorboard/") if self.args.savemodel else None,
                        device="auto",
                        policy_kwargs=onpolicy_kwargs
                        )
            print_ppo_conf(model)


        # tensorboard --logdir ./logs/ppo_tensorboard/

        elif self.args.agent == 'SAC':
            # vec_env = make_vec_env([make_env(gui=False, rank=i) for i in range(num_cpu)], n_envs=4, seed=0)
            # model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)

            model = SAC(
                "MlpPolicy",
                train_env,
                # replay_buffer_class=HerReplayBuffer,
                # replay_buffer_kwargs=dict(
                #     n_sampled_goal=len(self.targets),
                #     goal_selection_strategy="future",
                # ),
                verbose=0,
                tensorboard_log=(tensorboard_path + "/SAC_tensorboard/") if self.args.savemodel else None,
                train_freq=1,
                gradient_steps=2,
                buffer_size=int(1e6),
                learning_rate=int(self.args.learning_rate),
                # gamma=0.95,
                batch_size=self.args.batch_size,
                policy_kwargs=offpolicy_kwargs,  # dict(net_arch=[256, 256, 256]),
                device="auto",
            )
            print_sac_conf(model)

        elif self.args.agent == 'DDPG':
            model = DDPG("MlpPolicy",
                         train_env,
                         verbose=1,
                         batch_size=1024,
                         learning_rate=int(self.args.learning_rate),
                         train_freq=(10, "step"),
                         tensorboard_log=tensorboard_path + "/ddpg_tensorboard/",
                         device="auto",
                         policy_kwargs=dict(net_arch=[64, 64])
                         )

        model.set_random_seed(1)

        callbacks = []

        torch.distributions.Normal(0, 1).log_prob(torch.tensor(0.0))

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100_000, verbose=1)
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

        if self.args.savemodel:
            callbacks.append(Callbacks.FoundTargetsCallback(log_dir=chckpt_path + '/'))

        if self.args.wandb:
            callbacks.append(
                WandbCallback(gradient_save_freq=100, model_save_path=chckpt_path + "/wand/", verbose=2, ))

        callbacks.append(
            EvalCallback(eval_env,
                         # callback_on_new_best=callback_on_best,
                         best_model_save_path=chckpt_path + '/' if self.args.savemodel else None,
                         log_path=(chckpt_path + '/') if self.args.savemodel else None,
                         eval_freq=max(2000 // self.num_envs, 1),
                         n_eval_episodes=10,
                         deterministic=False,
                         render=False,
                         verbose=1,
                         )
        )
        # AimCallback(repo='.Aim/', experiment_name='sb3_test')

        trained_model = model.learn(total_timesteps=self.args.total_timesteps,
                    callback=callbacks,
                    log_interval=1000,
                    tb_log_name=self.args.agent + "_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                    )

        if self.args.savemodel:
            model.save(os.curdir + chckpt_path + '/success_model.zip')

            stats_path = os.path.join(chckpt_path, "vec_normalize.pkl")
            eval_env.save(stats_path)

            wandb.finish()

        # Saving the replay buffer
        # model.save_replay_buffer("sac_replay_buffer")
        #
        # Loading the replay buffer
        # model.load_replay_buffer("sac_replay_buffer")

        # Test the model #########################################
        if self.args.autotest:

            test_env = self.make_env(multi=False)
            test_env_nogui = self.make_env(multi=False)

            test_env = stable_baselines3.common.monitor.Monitor(test_env)
            test_env_nogui = stable_baselines3.common.monitor.Monitor(test_env_nogui)

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
                            output_folder=chckpt_path + "/logs/logger/" if self.args.savemodel else None,
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

        end = time.perf_counter()
        print(end - start)


#TODO
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: initial learning rate
    :return: schedule that computes the current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def lrsched():
    def reallr(progress):
        lr = 0.003
        if progress < 0.85:
            lr = 0.0005
        if progress < 0.66:
            lr = 0.00025
        if progress < 0.33:
            lr = 0.0001
        return lr

    return reallr
