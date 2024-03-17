import math
import os
import random
import time
from datetime import datetime
import argparse
from distutils.util import strtobool
# import sync, str2bool

import sys

sys.path.append("../")
sys.path.append("./")

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
from Sol.Model.SBActorCritic import CustomActorCriticPolicy

# from tf_agents.environments import py_environment

# import aim
from wandb.integration.sb3 import WandbCallback

th.autograd.set_detect_anomaly(True)

np.seterr(all="raise")

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_COLAB = False

register(
    # unique identifier for the env `name-version`
    id="DroneEnv",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="PBDroneSimulator",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=3000,
)


class PBDroneSimulator:
    def __init__(self, args, targets, target_factor=0,
                 plot=True,
                 discount=0.999,
                 threshold=0.3,
                 max_steps=5000,
                 num_cpu=24):

        self.plot = plot
        self.discount = discount
        self.threshold = args.threshold
        self.max_steps = args.max_steps

        # self.max_reward = 100 + len(targets) * 10

        self.num_envs = args.num_envs
        self.targets = self.dilate_targets(targets, target_factor)

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
                max_steps=self.max_steps,
                physics=Physics.PYB,
                gui=gui,
                initial_xyzs=initial_xyzs,
                save_folder=save_path,
                aviary_dim=aviary_dim,
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

        saved_filename = "C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.15.2023_17.09.00/best_model.zip"

        if args.agent == "SAC":
            model = SAC.load(saved_filename)
        if args.agent == "PPO":
            model = PPO.load(saved_filename)

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
                                            deterministic=False
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

    def test_learning(self):

        train_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i,
                                                 aviary_dim=np.array([-2, -2, 0, 2, 2, 2])) for i in
                                   range(1)])

        # custom_policy = CustomActorCriticPolicy(train_env.observation_space, train_env.action_space,
        #                                         net_arch=[512, 512, dict(vf=[256, 128],
        #                                                                  pi=[256, 128])],
        #                                         lr_schedule=linear_schedule(1e-3),
        #                                         activation_fn=th.nn.Tanh)

        onpolicy_kwargs = dict(net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])])

        model = PPO("MlpPolicy",
                    train_env,
                    verbose=1,
                    n_steps=2048,
                    batch_size=49,
                    device="auto",
                    policy_kwargs=onpolicy_kwargs
                    # dict(net_arch=[256, 256, 256], activation_fn=th.nn.GELU, ),
                    )
        print(model.policy)

        model.learn(total_timesteps=int(5e2), )

        print("#############################################")

        # model = PPO(custom_policy,
        #             train_env,
        #             verbose=1,
        #             n_steps=2048,
        #             batch_size=49
        #             # dict(net_arch=[256, 256, 256], activation_fn=th.nn.GELU, ),
        #             )

    def run_full(self):
        start = time.perf_counter()

        filename = os.path.join("./model_chkpts", 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename + '/')

        train_env = self.make_env(multi=False, gui=False)
        # check_env(train_env, warn=True, skip_render_check=True)

        train_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i,
                                                 aviary_dim=np.array([-2, -2, 0, 2, 2, 2])) for i in
                                   range(self.num_envs)])

        # eval_env = make_env(multi=False, gui=False, rank=0)

        eval_env = SubprocVecEnv([self.make_env(multi=True,
                                                save_path=filename if args.savemodel else None,
                                                gui=args.gui,
                                                aviary_dim=np.array([-2, -2, 0, 2, 2, 2])), ])
        # eval_env = SubprocVecEnv([self.make_env(multi=True, gui=False, rank=i) for i in range(self.num_cpu)])

        if args.vec_check_nan:
            train_env = VecCheckNan(train_env)
            eval_env = VecCheckNan(eval_env)

        if args.vec_normalize:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                                     clip_obs=1)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
                                    clip_obs=1)

        # On-policy algorithms #################################

        onpolicy_kwargs = dict(activation_fn=th.nn.Tanh,
                               share_features_extractor=True,
                               net_arch=dict(vf=[512, 512, 256, 128],
                                             pi=[512, 512, 256, 128]))

        custom_policy = dict(net_arch=[dict(share=[512, 512], vf=[256, 128], pi=[256, 128])],
                             activation_fn=th.nn.Tanh)

        # Off-policy algorithms #################################
        offpolicy_kwargs = dict(activation_fn=th.nn.ReLU,
                                net_arch=[512, 512, 256, 128])

        #     offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                             dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))

        if args.agent == 'PPO':
            model = PPO(ActorCriticPolicy,
                        train_env,
                        verbose=1,
                        n_steps=2048 * self.num_envs,
                        batch_size=49152 // self.num_envs,
                        ent_coef=0.01,
                        # use_sde=True,
                        # sde_sample_freq=4,
                        clip_range=0.2,
                        learning_rate=args.learning_rate,
                        tensorboard_log="./logs/ppo_tensorboard/" if args.savemodel else None,
                        device="auto",
                        policy_kwargs=onpolicy_kwargs
                        # dict(net_arch=[256, 256, 256], activation_fn=th.nn.ReLU, ),
                        )

        # tensorboard --logdir ./logs/ppo_tensorboard/

        elif args.agent == 'SAC':

            model = SAC(
                "MlpPolicy",
                train_env,
                # replay_buffer_class=HerReplayBuffer,
                # replay_buffer_kwargs=dict(
                #     n_sampled_goal=len(self.targets),
                #     goal_selection_strategy="future",
                # ),
                verbose=0,
                tensorboard_log="./logs/SAC_tensorboard/" if args.savemodel else None,
                train_freq=1,
                gradient_steps=2,
                buffer_size=int(1e6),
                learning_rate=args.learning_rate,
                # gamma=0.95,
                batch_size=49152 // self.num_envs,
                policy_kwargs=offpolicy_kwargs,  # dict(net_arch=[256, 256, 256]),
                device="auto",
            )

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

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100_000, verbose=1)

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

        found_tar_callback = Callbacks.FoundTargetsCallback(log_dir=filename + '/')

        wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=filename + '/' + "wand", verbose=2, )

        eval_callback = EvalCallback(eval_env,
                                     # callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path=filename + '/' if args.savemodel else None,
                                     log_path=filename + '/' if args.savemodel else None,
                                     eval_freq=int(2000 / self.num_envs),
                                     deterministic=False,
                                     render=False
                                     )

        model.learn(total_timesteps=int(args.max_steps),
                    callback=[eval_callback,
                              found_tar_callback if args.savemodel else None,
                              wandb_callback if args.wandb else None,
                              # AimCallback(repo='.Aim/', experiment_name='sb3_test')
                              ],
                    log_interval=1000,
                    tb_log_name=args.agent + " " + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                    )

        if args.savemodel:
            model.save(os.curdir + filename + '/success_model.zip')

            stats_path = os.path.join(filename, "vec_normalize.pkl")
            eval_env.save(stats_path)

            wandb.finish()

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

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument('--gym_id', type=str, default='PBDroneEnv',
                        help="the id of the gym environment")
    parser.add_argument('--run_type', type=str, default='full', choices=["full", "test", "saved", "learning"])
    parser.add_argument('--env-config', type=str, default='default')
    parser.add_argument('--env-kwargs', type=str, default='{}')

    parser.add_argument('--seed', '-s', type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument('--gui', default=DEFAULT_GUI, help='Whether to use PyBullet GUI for the eval env'
                                                           '(default: False)')

    # Saving
    parser.add_argument('--savemodel', type=bool, default=True)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--savedir', type=str, default='')
    parser.add_argument('--checkpoint-freq', type=int, default=100)

    # Wrapper specific arguments
    parser.add_argument('--vec_check_nan', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--vec_normalize', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--ve_check_env', default=False, type=lambda x: bool(strtobool(x)))

    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument('--max_steps', type=int, default=5e6,
                        help="total timesteps of the experiments")
    parser.add_argument('--max_env_steps', type=int, default=5000,
                        help="total timesteps of one episode")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="the learning rate of the optimizer")

    # RL Algorithm specific arguments
    parser.add_argument('--agent', type=str, default='PPO')
    parser.add_argument('--agent-config', type=str, default='default')
    parser.add_argument('--discount', type=int, default=0.999)
    parser.add_argument('--threshold', type=int, default=0.3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-steps', type=int, default=2048)

    # PPO specific
    parser.add_argument('--clip_range', type=int, default=0.2)
    parser.add_argument('--ent_coef', type=int, default=0)

    parser.add_argument('--eval-criterion', type=str, default='default')
    parser.add_argument('--eval-criterion-config', type=str, default='default')
    parser.add_argument('--metric', type=str, default='default')
    parser.add_argument('--metric-config', type=str, default='default')
    parser.add_argument('--optimizer', type=str, default='default')
    parser.add_argument('--optimizer-config', type=str, default='default')
    parser.add_argument('--criterion', type=str, default='default')
    parser.add_argument('--criterion-config', type=str, default='default')

    # Wandb specific arguments
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--wandb_rootlog', type=str, default="/wandb")

    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    args = parser.parse_args()

    return args


def init_wandb(args):
    run_name = f"{args.gym_id}__{args.agent}__{int(time.time())}"
    print(f"Starting run {run_name} with `wandb`...")

    # wandb.tensorboard.patch(root_logdir=args.wandb_rootlog)

    run = wandb.init(
        project="rl",
        config=args,
        name=run_name,
        tensorboard=True,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional

    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


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


if __name__ == "__main__":
    # C:\Users\xx4qw\anaconda3\envs\CondaDrone\python.exe C:\Files\Egyetem\Szakdolgozat\RL\Sol\Model\simulation_controller.py --agent SAC

    args = parse_args()
    print(args)
    #

    ## seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = False
    #
    # device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # targets = Waypoints.up_circle()
    targets = Waypoints.rnd()

    sim = PBDroneSimulator(args, targets, target_factor=0)

    if args.wandb:
        init_wandb(args)

    if args.run_type == "full":
        sim.run_full()
    elif args.run_type == "test":
        sim.run_test()
    elif args.run_type == "saved":
        sim.test_saved()
    elif args.run_type == "learning":
        sim.test_learning()

    # video_recorder.record_video(
    #     model=PPO.load("C:\Files\Egyetem\Szakdolgozat\RL\Sol\model_chkpts\save-12.04.2023_22.26.05/best_model.zip",
    #                    video_folder="C:\Files\Egyetem\Szakdolgozat\RL\Sol/results/videos",
    #                    ))
