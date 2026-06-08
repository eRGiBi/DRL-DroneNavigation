import os
import sys
import time
import random
import logging
from datetime import datetime

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
from torch.utils.tensorboard import SummaryWriter

import wandb
# import aim

from gymnasium.envs.registration import register

from Sol.Model.PBDroneSimulator import PBDroneSimulator
import Sol.Utilities.waypoints as Waypoints
from Sol.Utilities.waypoints import Track
from Sol.Utilities.arg_parser import parse_args
from Sol.Utilities.Profiler import Profiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

def main():

    init_env()

    args = parse_args()

    print('Initial parsed arguments:')
    print(args)

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    # th.backends.cudnn.deterministic = True

    args.device = th.device("cuda" if th.cuda.is_available() and args.device == "cuda" else "cpu")
    args.now = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

    # Track selection
    track = Track(Waypoints.circle(radius=1, num_points=6, height=1), circle=True)
    # track = Track(Waypoints.reaching())
    # track = Track(Waypoints.up_sharp_back_turn())
    # track = Track(Waypoints.up_circle(), circle=False)

    sim = PBDroneSimulator(args, track, target_factor=0)

    if args.run_type == "full" or args.run_type == "cont":

        if args.wandb:
            init_wandb(args)

        try:
            # Setup logging
            if args.profile:

                profiler = Profiler()

                with profiler:
                    sim.run_full_training()
                    profiler.print_stats()

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


if __name__ == "__main__":

    main()
