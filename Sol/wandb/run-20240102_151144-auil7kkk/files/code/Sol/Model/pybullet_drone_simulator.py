import math
import os
import random
import time
from datetime import datetime
import argparse
from distutils.util import strtobool
# import sync, str2bool

from typing import Callable

import gym.wrappers
import numpy as np
import torch as th
import wandb

from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecCheckNan, VecNormalize, VecTransposeImage

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO, SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter

from Sol.PyBullet.FlyThruGateAviary import FlyThruGateAviary
# from PyBullet import BaseAviary
from Sol.PyBullet.enums import Physics
# from Sol.DroneEnvironment import DroneEnvironment
from Sol.Model.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.Logger import Logger
import Sol.Model.Waypoints as Waypoints

from Sol.Utilities.Plotter import plot_learning_curve, plot_metrics, plot_3d_targets
import Sol.Utilities.Callbacks as Callbacks

# from tf_agents.environments import py_environment

# import aim
from wandb.integration.sb3 import WandbCallback

th.autograd.set_detect_anomaly(True)

np.seterr(all="raise")

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

register(
    # unique identifier for the env `name-version`
    id="DroneEnv",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="PBDroneEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=3000,
)


class PBDroneSimulator:
    def __init__(self, targets, target_factor=0,
                 plot=True,
                 discount=0.999,
                 threshold=0.3,
                 max_steps=5000,
                 num_cpu=24):

        self.plot = plot
        self.discount = discount
        self.threshold = threshold
        self.max_steps = max_steps

        # self.max_reward = 100 + len(targets) * 10

        self.num_cpu = num_cpu
        self.targets = self.dilate_targets(targets, target_factor)

    def make_env(self, multi=False, gui=False, initial_xyzs=None, aviary_dim=np.array([-1, -1, 0, 1, 1, 1]),
                 rank: int = 0, seed: int = 0,
                 save_model: bool = False, save_path: str = None):
        """
        Utility function for multiprocessed env.
        """

        def _init():
            env = PBDroneEnv(
                target_points=self.targets,
                threshold=self.threshold,
                discount=self.discount,
                max_steps=self.max_steps,
                physics=Physics.PYB,
                gui=gui,
                initial_xyzs=initial_xyzs,
                save_model=save_model,
                save_folder=save_path,
                aviary_dim=aviary_dim,
            )
            env.reset(seed=seed + rank)
            env = Monitor(env) # record stats such as returns
            return env

        if multi:
            set_random_seed(seed)
            return _init
        else:
            return _init()

    def dilate_targets(self, targets, factor: int) -> list:

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

    def run_test(self):
        # action = np.array([-.1, -.1, -.1, -.1], dtype=np.float32)
        # action = np.array([-.9, -.9, -.9, -.9], dtype=np.float32)
        # action = np.array([.9, .9, .9, .9], dtype=np.float32)
        action = np.array([-1, -1, -1, -1], dtype=np.float32)
        action *= -1
        # action = np.array([0, 0, 0, 0], dtype=np.float32)
        plot_3d_targets(self.targets)

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

        model = SAC.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.29.2023_11.55.59/best_model.zip")
        # model = PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\sa/best_model.zip",
        #                  env=drone_environment)
        # model = PPO.load(os.curdir + "\model_chkpts\success_model.zip")
        # model = SAC.load(os.curdir + "\model_chkpts\success_model.zip")

        rewards = []
        rewards_sum = []
        images = []
        obs, info = drone_environment.reset()

        # drone_environment = gym.wrappers.RecordEpisodeStatistics(drone_environment)

        for target in self.targets:
            print(target)

        #### Print training progression ############################
        #     with np.load(filename+'/evaluations.npz') as data:
        #         for j in range(data['timesteps'].shape[0]):
        #             print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

        for i in range(self.max_steps):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = drone_environment.step(action)
            print(i)
            print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
                  truncated)
            print("rpy", drone_environment.rpy)
            print("pos", drone_environment.pos[0])
            rewards.append(reward)
            rewards_sum.append(sum(rewards))
            # images.append(img)

            if terminated:
                plot_learning_curve(rewards)
                plot_learning_curve(rewards_sum, title="Cumulative Rewards")
                print("cum", sum(rewards))

                # imageio.mimsave("save-12.03.2023_20.10.04.gif",
                #                 [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
                #                 duration=0.58*(1000 * 1/50))

                drone_environment.reset()

            time.sleep(1. / 240.)

    def run_full(self, args):
        start = time.perf_counter()

        filename = os.path.join("./model_chkpts", 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename + '/')

        train_env = self.make_env(multi=False, gui=False)
        # check_env(train_env, warn=True, skip_render_check=True)

        train_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i,
                                                 aviary_dim=np.array([-2, -2, 0, 2, 2, 2])) for i in
                                   range(self.num_cpu)])
        # train_env = VecCheckNan(train_env)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                 clip_obs=1)

        # eval_env = make_env(multi=False, gui=False, rank=0)
        #
        eval_env = SubprocVecEnv([self.make_env(multi=True, save_model=True, save_path=filename, gui=False,
                                                aviary_dim=np.array([-2, -2, 0, 2, 2, 2])), ])
        # eval_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i) for i in range(self.num_cpu)])
        # eval_env = VecCheckNan(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
                                clip_obs=1)

        # onpolicy_kwargs = dict(activation_fn=th.nn.ReLU,
        #                        net_arch=dict(vf=[512, 512, 256, 128],
        #                                      pi=[512, 512, 256, 128]))
        # onpolicy_kwargs = dict(net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])])

        model = PPO("MlpPolicy",
                    train_env,
                    verbose=1,
                    n_steps=2048,
                    batch_size=49152,
                    ent_coef=0.01,
                    # use_sde=True,
                    # sde_sample_freq=4,
                    clip_range=0.2,
                    learning_rate=1e-3,
                    tensorboard_log="./logs/ppo_tensorboard/",
                    device="auto",
                    policy_kwargs=  # onpolicy_kwargs
                    dict(net_arch=[256, 256, 256], activation_fn=th.nn.Tanh, ),
                    )

        # tensorboard --logdir ./logs/ppo_tensorboard/

        #### Off-policy algorithms #################################
        #     offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                             net_arch=[512, 512, 256, 128]

        #     offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                             dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))

        # model = SAC(
        #     "MultiInputPolicy",
        #     train_env,
        #     replay_buffer_class=HerReplayBuffer,
        #     replay_buffer_kwargs=dict(
        #         n_sampled_goal=len(self.targets),
        #         goal_selection_strategy="future",
        #     ),
        #     verbose=0,
        #     tensorboard_log="./logs/SAC_tensorboard/",
        #     train_freq=1,
        #     gradient_steps=2,
        #     buffer_size=int(3e6),
        #     learning_rate=1e-3,
        #     # gamma=0.95,
        #     batch_size=2048,
        #     policy_kwargs=dict(net_arch=[256, 256, 256]),
        #     device="auto",
        # )
        # train_env = make_vec_env(make_env(multi=False), n_envs=12)

        # model = DDPG("MlpPolicy",
        #              train_env,
        #              verbose=1,
        #              batch_size=1024,
        #              learning_rate=1e-3,
        #              train_freq=(10, "step"),
        #              tensorboard_log="./logs/ddpg_tensorboard/",
        #              device="auto",
        #              policy_kwargs=dict(net_arch=[64, 64])
        #              )
        model.set_random_seed(1)

        # vec_env = make_vec_env([make_env(gui=False, rank=i) for i in range(num_cpu)], n_envs=4, seed=0)
        # model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)

        # train_env = stable_baselines3.common.monitor.Monitor(train_env)
        # eval_env = stable_baselines3.common.monitor.Monitor(eval_env)

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100_000,
                                                         verbose=1)
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

        found_tar_callback = Callbacks.FoundTargetsCallback(log_dir=filename + '/')

        wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=filename + '/', verbose=2,)

        eval_callback = EvalCallback(eval_env,
                                     # callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path=filename + '/',
                                     log_path=filename + '/',
                                     eval_freq=int(2000 / self.num_cpu),
                                     deterministic=False,
                                     render=False
                                     )

        model.learn(total_timesteps=int(1e7),
                    callback=[eval_callback,
                              found_tar_callback
                              # AimCallback(repo='.Aim/', experiment_name='sb3_test')
                              ],
                    log_interval=1000,
                    )

        model.save(os.curdir + filename + '/success_model.zip')

        stats_path = os.path.join(filename, "vec_normalize.pkl")
        eval_env.save(stats_path)

        # Test the model #########################################

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
                        output_folder=os.curdir + "/logs"
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


def parse_args():
    """Parse arguments from the command line."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-id', type=str, default='PBDroneEnv')
    parser.add_argument('--env', type=str, default='default')
    parser.add_argument('--env-config', type=str, default='default')
    parser.add_argument('--env-kwargs', type=str, default='{}')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--gui', default=DEFAULT_GUI, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--save-buffer', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--save-obs', action='store_true', default=False)
    parser.add_argument('--save-video', action='store_true', default=False)
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--checkpoint-freq', type=int, default=100)
    parser.add_argument('--checkpoint-at-end', action='store_true', default=False)
    parser.add_argument('--restore-agent', action='store_true', default=False)
    parser.add_argument('--restore-buffer', action='store_true', default=False)
    parser.add_argument('--restore-optimizer', action='store_true', default=False)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=5e6)
    parser.add_argument('--agent', type=str, default='PPO')
    parser.add_argument('--agent-config', type=str, default='default')
    parser.add_argument('--policy', type=str, default='default')
    parser.add_argument('--policy-config', type=str, default='default')
    parser.add_argument('--eval-criterion', type=str, default='default')
    parser.add_argument('--eval-criterion-config', type=str, default='default')
    parser.add_argument('--metric', type=str, default='default')
    parser.add_argument('--metric-config', type=str, default='default')
    parser.add_argument('--optimizer', type=str, default='default')
    parser.add_argument('--optimizer-config', type=str, default='default')
    parser.add_argument('--criterion', type=str, default='default')
    parser.add_argument('--criterion-config', type=str, default='default')

    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # #### Define and parse (optional) arguments for the script ##
    # parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    # parser.add_argument('--gui', default=DEFAULT_GUI, help='Whether to use PyBullet GUI (default: True)',
    #                     metavar='')
    # parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
    #                     help='Whether to record a video (default: False)', metavar='')
    # parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
    #                     help='Folder where to save logs (default: "results")', metavar='')
    # parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
    #                     help='Whether example is being run by a notebook (default: "False")', metavar='')
    #
    # run(**vars(ARGS))

    args = parser.parse_args()

    return args


def init_wandb(args):

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(f"Starting run {run_name} with `wandb`...")

    config = {
        "env_name": args.env,
        "policy_type": args.policy,
        "total_timesteps": args.num_steps,
        "policy_config": args.policy_config,
        "env_config": args.env_config,
        "seed": args.seed,
        "agent": args.agent,
        "agent_config": args.agent_config,
        "metric": args.metric,
        "metric_config": args.metric_config,
        "optimizer": args.optimizer,
        "optimizer_config": args.optimizer_config,
        "criterion": args.criterion,
    }
    run = wandb.init(
        project="rl",
        config=args,
        name=args.exp_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


if __name__ == "__main__":

    args = parse_args()
    print(args)
    #

    # ## seeding
    # seed = args.seed
    # random.seed(seed)
    # np.random.seed(seed)
    # th.manual_seed(seed)
    # th.backends.cudnn.deterministic = False
    #
    # device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # targets = Waypoints.up_circle()
    targets = Waypoints.rnd()

    sim = PBDroneSimulator(targets, target_factor=0)

    init_wandb(args)

    sim.run_full(args)
    #
    # sim.run_test()

    # sim.test_saved()
    #

    # video_recorder.record_video(
    #     model=PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.04.2023_22.26.05/best_model.zip",
    #                    video_folder="C:\Files\Egyetem\Szakdolgozat\RL\Sol/results/videos",
    #                    ))


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



def manual_pb_env():
    # Connect to the PyBullet physics server
    # physicsClient = p.connect(p.GUI)
    # p.setGravity(0, 0, -9.81)
    # p.setRealTimeSimulation(0)
    # Load the drone model
    # drone = p.loadURDF("cf2x.urdf", [0, 0, 0])

    print("----------------------------")
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
