import os
import time
from datetime import datetime
# import sync, str2bool

from typing import Callable

import base64
import pathlib
from pathlib import Path

import imageio
from IPython import display as ipythondisplay

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register

import stable_baselines3.common.monitor

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, A2C, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.results_plotter import load_results, ts2xy

from stable_baselines3.common.utils import set_random_seed

# from PyBullet import BaseAviary
from PyBullet.enums import Physics
from Sol.DroneEnvironment import DroneEnvironment
from Sol.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.Logger import Logger
import Sol.Utilities.video_recorder as video_recorder

# from tf_agents.environments import py_environment

import torch as th
th.autograd.set_detect_anomaly(True)

np.seterr(all="raise")

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

INITIAL_POS = np.array([[0, 0, 0]])
plot = True
discount = 0.999
threshold = 0.05
max_steps = 5000

num_cpu = 4

targets = [np.array([0.0, 0.0, 0.1]),
           np.array([0.0, 0.0, 0.2]),
           np.array([0., 0., 0.3]),
           np.array([0., 0., 0.4]),
           np.array([0., 0., 0.5]),
           np.array([0., 0.1, 0.5]),
           np.array([0., 0.2, 0.5]),
           np.array([0., 0.35, 0.5]),
           np.array([0., 0.5, 0.5]),
           np.array([0.1, 0.5, 0.5]),
           np.array([0.25, 0.5, 0.5]),
           np.array([0.25, 0.5, 5]),
           np.array([0.5, 0.5, 5]),
           # np.array([1., .1, .]),
           # np.array([1., 1., 1.]),
           ]

# targets = [np.array([0.0, 0.0, 0.5]),
#            np.array([0.0, 0.5, 0.5]),
#            np.array([0.25, 0.25, 0.25]),
#            ]

max_reward = 100 + len(targets) * 10

register(
    # unique identifier for the env `name-version`
    id="DroneEnv",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="PBDroneEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=3000,
)


def plot_learning_curve(scores, title='Learning Curve'):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Total Reward per Episode')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(loc='upper left')
    plt.show()


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


def make_env(multi=True, gui=False, rank: int = 0, seed: int = 0, ):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = PBDroneEnv(
            target_points=targets,
            threshold=threshold,
            discount=discount,
            max_steps=max_steps,
            physics=Physics.PYB,
            gui=gui,
            initial_xyzs=INITIAL_POS,
        )
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    if multi:
        set_random_seed(seed)
        return _init
    else:
        return _init()



def run_test():
    action = np.array([-.1, -.1, -.1, -.1], dtype=np.float32)
    action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
    action = np.array([.9, .9, .9, .9], dtype=np.float32)
    # action * -1

    drone_environment = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        max_steps=max_steps,
        physics=Physics.PYB,
        gui=True,
        initial_xyzs=np.array([[0, 0, 0]], dtype=np.float64),
    )

    # It will check your custom environment and output additional warnings if needed
    check_env(drone_environment, warn=True)

    print('[INFO] Action space:', drone_environment.action_space)
    print('[INFO] Observation space:', drone_environment.observation_space)

    rewards = []
    rewards_sum = []
    time_step = drone_environment.reset()
    print(time_step)

    # for _ in range(100):
    while True:
        time_step = drone_environment.step(action)
        rewards.append(time_step[1])
        rewards_sum.append(sum(rewards))
        print(time_step)
        print(drone_environment._steps)
        if time_step[2]:
            break

        time.sleep(1. / 240.)  # Control the simulation speed

    plot_learning_curve(rewards)
    plot_learning_curve(rewards_sum)


def test_saved():
    drone_environment = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        max_steps=max_steps,
        physics=Physics.PYB,
        record=True,
        vision_attributes=True,
        gui=True,
        initial_xyzs=np.array([[0, 0, 0]])
    )
    # model = SAC.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.05.2023_17.41.00/best_model.zip")
    model = PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.10.2023_15.56.55/best_model.zip",
                     env=drone_environment)
    # model = PPO.load(os.curdir + "\model_chkpts\success_model.zip")
    # model = SAC.load(os.curdir + "\model_chkpts\success_model.zip")

    rewards = []
    rewards_sum = []
    images = []
    obs, info = drone_environment.reset()

    drone_environment._startVideoRecording()

    for i in range(5000):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = drone_environment.step(action)
        print(i)
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
              truncated)
        print(drone_environment.rpy)
        print(drone_environment.pos[0])
        rewards.append(reward)
        rewards_sum.append(sum(rewards))
        # images.append(img)

        if terminated:
            plot_learning_curve(rewards)
            plot_learning_curve(rewards_sum, title="Cumulative Rewards")

            # imageio.mimsave("save-12.03.2023_20.10.04.gif",
            #                 [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
            #                 duration=0.58*(1000 * 1/50))

            drone_environment.reset()

        time.sleep(1. / 240.)


def run_full():
    start = time.perf_counter()

    filename = os.path.join("model_chkpts", 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    # Connect to the PyBullet physics server
    # physicsClient = p.connect(p.GUI)
    # p.setGravity(0, 0, -9.81)
    # p.setRealTimeSimulation(0)
    # Load the drone model
    # drone = p.loadURDF("cf2x.urdf", [0, 0, 0])

    # print("----------------------------")
    # print(drone_environment.action_space)
    # print(drone_environment.action_spec())
    # print(drone_environment.getDroneIds())
    # print(drone_environment.observation_spec())
    # print("----------------------------")

    # tf_env = tf_py_environment.TFPyEnvironment(drone_environment)
    #
    # print('action_spec:', tf_env.action_spec())
    # print('time_step_spec.observation:', tf_env.time_step_spec().observation)
    # print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
    # print('time_step_spec.discount:', tf_env.time_step_spec().discount)
    # print('time_step_spec.reward:', tf_env.time_step_spec().reward)

    # train_env = make

    train_env = SubprocVecEnv([make_env(gui=False, rank=i) for i in range(num_cpu)])
    # train_env = VecCheckNan(train_env)

    # eval_env = make_env(gui=False, rank=0)

    eval_env = SubprocVecEnv([make_env()])
    # eval_env = VecCheckNan(eval_env)

    model = PPO("MlpPolicy",
                train_env,
                verbose=1,
                tensorboard_log="./logs/ppo_tensorboard/",
                batch_size=256,
                gamma=discount,
                device="cuda",
                )
    # tensorboard --logdir ./logs/ppo_tensorboard/

    # model = SAC(
    #     "MlpPolicy",
    #     train_env,
    #     # replay_buffer_class=HerReplayBuffer,
    #     # replay_buffer_kwargs=dict(
    #     #     n_sampled_goal=len(targets),
    #     #     goal_selection_strategy="future",
    #     # ),
    #     verbose=1,
    #     tensorboard_log="./logs/SAC_tensorboard/",
    #     train_freq=1, gradient_steps=2,
    #     # buffer_size=int(1e6),
    #     # learning_rate=1e-3,
    #     # gamma=0.95,
    #     # batch_size=256,
    #     # policy_kwargs=dict(net_arch=[256, 256, 256]),
    # )

    # vec_env = make_vec_env([make_env(gui=False, rank=i) for i in range(num_cpu)], n_envs=4, seed=0)
    # model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)

    # train_env = stable_baselines3.common.monitor.Monitor(train_env)
    # eval_env = stable_baselines3.common.monitor.Monitor(eval_env)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20000,
                                                     verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(200),
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=5_000_000,
                callback=eval_callback,
                log_interval=1000,
                )

    model.save(os.curdir + filename + '/success_model.zip')
    rewards = []

    # if os.path.isfile(filename + '/success_model.zip'):
    #     path = filename + '/success_model.zip'
    # elif os.path.isfile(filename + '/best_model.zip'):
    #     path = filename + '/best_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # model = PPO.load(path)

    train_env.close()

    test_env = make_env(multi=False)
    test_env_nogui = make_env(multi=False)

    # test_env = stable_baselines3.common.monitor.Monitor(test_env)
    # test_env_nogui = stable_baselines3.common.monitor.Monitor(test_env_nogui)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=os.curdir + "/logs"
                    )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=100
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42)
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

    if plot:
        logger.plot()

    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    # vec_env = SubprocVecEnv([make_env(gui=False, rank=i) for i in range(num_cpu)])
    #
    run_full()

    # run_test()
    #
    # test_saved()
    #


    # video_recorder.record_video(
    #     model=PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.04.2023_22.26.05/best_model.zip",
    #                    video_folder="C:\Files\Egyetem\Szakdolgozat\RL\Sol/results/videos",
    #                    ))


#     #### Define and parse (optional) arguments for the script ##
#     parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
#     parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
#                         metavar='')
#     parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
#                         help='Whether to record a video (default: False)', metavar='')
#     parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
#                         help='Folder where to save logs (default: "results")', metavar='')
#     parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
#                         help='Whether example is being run by a notebook (default: "False")', metavar='')
#     ARGS = parser.parse_args()
#
#     run(**vars(ARGS))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

# # now save the replay buffer too
# model.save_replay_buffer("sac_replay_buffer")
#
# # load it into the loaded_model
# loaded_model.load_replay_buffer("sac_replay_buffer")


# import imageio
# import numpy as np
#
# from stable_baselines3 import A2C
#
# model = A2C("MlpPolicy", "LunarLander-v2").learn(100_000)
#
# images = []
# obs = model.env.reset()
# img = model.env.render(mode="rgb_array")
# for i in range(350):
#     images.append(img)
#     action, _ = model.predict(obs)
#     obs, _, _ ,_ = model.env.step(action)
#     img = model.env.render(mode="rgb_array")
#
# imageio.mimsave("lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
