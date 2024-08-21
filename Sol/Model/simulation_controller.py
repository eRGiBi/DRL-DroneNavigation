import logging
import os
import random
import sys
import time

import cProfile
import pstats

# TODO
sys.path.append("../")
sys.path.append("./")
sys.path.append("Sol/Model")
sys.path.append("Sol.Model")

# Get the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '.'))

# Add the root directory to the sys.path
sys.path.append(root_dir)

import numpy as np
import torch as th
import wandb
# import aim

from Sol.Model.PBDroneSimulator import PBDroneSimulator
import Sol.Utilities.Waypoints as Waypoints
from Sol.Utilities.ArgParser import parse_args
from Sol.Utilities.Profiler import Profiler

from gymnasium.envs.registration import register

from torch.utils.tensorboard import SummaryWriter


def init_env():
    th.autograd.set_detect_anomaly(True)

    np.seterr(all="raise")

    register(
        id="DroneEnv",
        entry_point="PBDroneSimulator",
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
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,

    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


if __name__ == "__main__":

    init_env()

    args = parse_args()

    # if args.yaml:
    #     with open('parameters.yml', 'r') as file:
    #         params = yaml.safe_load(file)

    print(args)

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    # th.backends.cudnn.deterministic = True

    # device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    track = Waypoints.Track(Waypoints.circle(radius=1, num_points=6, height=1), circle=True)
    # track = Waypoints.Track(Waypoints.reaching())
    # track = Waypoints.Track(Waypoints.up_sharp_back_turn())
    # track = Waypoints.Track(Waypoints.up_circle(), circle=False)

    sim = PBDroneSimulator(args, track, target_factor=0)

    if args.wandb:
        init_wandb(args)

    if args.run_type == "full" or args.run_type == "cont":

        try:
            # Setup logging
            if args.profile:
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                profiler = Profiler()

                with profiler:
                    sim.run_full_training()

            else:
                sim.run_full_training()

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        except Exception as e:
            print("Exception Occurred")
            print(e)
        finally:
            wandb.finish()

    elif args.run_type == "test":
        sim.run_test()
    elif args.run_type == "saved":
        sim.test_saved()
    elif args.run_type == "learning":
        sim.test_learning()

    # Manual PyBullet Environment for debugging #################################

    # def manual_pb_env():

    # physicsClient = p.connect(p.GUI)
    # p.setGravity(0, 0, -9.81)
    # p.setRealTimeSimulation(0)
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
