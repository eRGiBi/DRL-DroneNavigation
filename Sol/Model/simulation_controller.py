import math
import os
import random
import sys

import time
from datetime import datetime

from typing import Callable

# TODO
sys.path.append("../")
sys.path.append("./")

import numpy as np
import torch as th
import wandb
# import aim

from Sol.Model.PBDroneSimulator import PBDroneSimulator
import Sol.Model.Waypoints as Waypoints
from Sol.Utilities.ArgParser import parse_args

from gymnasium.envs.registration import register
import gym.wrappers

from torch.utils.tensorboard import SummaryWriter

# from tf_agents.environments import py_environment


def init_env():
    th.autograd.set_detect_anomaly(True)

    np.seterr(all="raise")

    register(
        # unique identifier for the env `name-version`
        id="DroneEnv",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="PBDroneSimulator",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=3000,
    )


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

    init_env()

    args = parse_args()
    print(args)

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = False

    # device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # targets = Waypoints.up_circle()
    targets = Waypoints.forward_up()
    # targets = Waypoints.half_up_forward()

    sim = PBDroneSimulator(args, targets, target_factor=0)

    if args.wandb:
        init_wandb(args)

    if args.run_type == "full":
        sim.run_full_training()
    elif args.run_type == "test":
        sim.run_test()
    elif args.run_type == "saved":
        sim.test_saved()
    elif args.run_type == "learning":
        sim.test_learning()
