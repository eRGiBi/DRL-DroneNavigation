import pathlib
import time
from datetime import datetime
# import sync, str2bool
import os

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

# from PyBullet import BaseAviary
from PyBullet.enums import Physics
from Sol.DroneEnvironment import DroneEnvironment
from Sol.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.Logger import Logger

# from tf_agents.environments import py_environment

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

plot = True

discount = 0.999
threshold = 0.1

targets = [np.array([0.0, 0.0, .5]),
           np.array([0., 0., 0.2]),
           # np.array([0., 0., 0.0]),
           # np.array([0., 0.1, 1.]),
           # np.array([1., .1, 0.]),
           # np.array([1., 1., 1.]),
           ]

max_reward = 100 + len(targets) * 10


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



def run_test():
    action = np.array([-.1, -.1, -.1, -.1], dtype=np.float32)
    action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
    action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
    action * -1

    drone_environment = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        physics=Physics.PYB,
        gui=True,
        initial_xyzs=np.array([[0, 0, 1]]),
    )
    rewards = []
    time_step = drone_environment.step(action / 100)
    print(time_step)

    # for _ in range(100):
    while True:
        time_step = drone_environment.step(action)
        rewards.append(time_step[1])
        print(time_step)
        if time_step[2]:
            break

        time.sleep(1. / 740.)  # Control the simulation speed

    plot_learning_curve(rewards)

def test_saved():
    model = PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.02.2023_22.13.02/best_model.zip")

    drone_environment = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        physics=Physics.PYB,
        gui=True,
        initial_xyzs=np.array([[0, 0, 0]]),
    )
    obs, info = drone_environment.reset(seed=42)
    while True:
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = drone_environment.step(action)
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
              truncated)

        if terminated:
            drone_environment.reset()


def run_full():
    start = time.perf_counter()

    filename = os.path.join("model_chkpts", 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    # env = e.MinitaurBulletEnv(render=True)

    # def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB,
    #         record_video=DEFAULT_RECORD_VIDEO):
    #     env = DroneEnvironment()
    #
    #     model = ()
    #
    #     #### Show (and record a video of) the model's performance ##
    #     env = FlyThruGateAviary(gui=gui,
    #                             record=record_video
    #                             )
    #     logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
    #                     num_drones=1,
    #                     output_folder=output_folder,
    #                     colab=colab
    #                     )
    #
    #     obs, info = env.reset(seed=42, options={})
    #     start = time.time()
    #
    #     for i in range(3 * env.CTRL_FREQ):
    #
    #         action, _states = model.predict(obs, deterministic=True)
    #
    #         obs, reward, terminated, truncated, info = env.step(action)
    #
    #         logger.log(drone=0,
    #                    timestamp=i / env.CTRL_FREQ,
    #                    state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
    #                    control=np.zeros(12)
    #                    )
    #         env.render()
    #         print(terminated)
    #         sync(i, start, env.CTRL_TIMESTEP)
    #         if terminated:
    #             obs = env.reset(seed=42, options={})
    #     env.close()
    #
    #     if plot:
    #         logger.plot()

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

    train_env = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        gui=True,
        initial_xyzs=np.array([[0, 0, 0]]),
    )
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    eval_env = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        physics=Physics.PYB,
        gui=False,
        initial_xyzs=np.array([[0, 0, 0]])
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1
    )

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=max_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=1_000_000,
                callback=eval_callback,
                log_interval=100)

    model.save(os.curdir + "/model_chkpts" + '/success_model.zip')
    rewards = []
    #
    # obs, info = drone_environment.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = drone_environment.step(action)
    #     print(obs)
    #     print(reward)
    #     rewards.append(reward)
    #     if terminated or truncated:
    #         obs, info = drone_environment.reset()
    #         print(rewards)
    #         print(sum(rewards))
    #         rewards = []
    #         break

    #

    # if os.path.isfile(filename + '/success_model.zip'):
    #     path = filename + '/success_model.zip'
    # elif os.path.isfile(filename + '/best_model.zip'):
    #     path = filename + '/best_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # model = PPO.load(path)

    train_env.close()

    test_env = PBDroneEnv(
        target_points=targets,
        threshold=discount,
        discount=threshold,
        physics=Physics.PYB,
        gui=True,
        initial_xyzs=np.array([[0, 0, 0]]),
        record=True
    )
    test_env_nogui = PBDroneEnv(
        target_points=targets,
        threshold=threshold,
        discount=discount,
        physics=Physics.PYB,
        gui=False,
        initial_xyzs=np.array([[0, 0, 0]]),
    )
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=os.curdir + "/logs"
                    )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42)
    start = time.time()
    print("wtasdas", (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        print("act", action)
        print("state", _states)
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

        # test_env.render()
        #         print(terminated)
        #         sync(i, start, test_env.CTRL_TIMESTEP)
        #         if terminated:
        #             obs = test_env.reset(seed=42, options={})

    test_env.close()

    if plot:
        logger.plot()

    plot_learning_curve(rewards)
    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    # run_full()

    run_test()

    # test_saved()


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


