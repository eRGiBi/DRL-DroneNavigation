import argparse
import os
from distutils.util import strtobool
from Sol.Model.parameter_directory.parameter_manager import *

def parse_args():
    """Parse arguments from the command line."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument('--gym_id', type=str, default='PBDroneEnv',
                        help="The id of the gym environment.")
    parser.add_argument('--lib', type=str, default='sb3', choices=["sb3", "ray", "tfa", "clrl"])
    parser.add_argument('--run_type', type=str, default='full',
                        choices=["full", "cont", "test", "saved", "learning"])
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--seed', '-s', type=int, default=gen_params['seed'], help="Seed of the experiment.")
    parser.add_argument('--gui', default=False, help='Whether to use PyBullet GUI for the eval env.',
                        type=lambda x: bool(strtobool(x)))

    parser.add_argument('--obs', type=str, default="pos", choices=["pos", "pos_ext", "rgb"])
    parser.add_argument('--profile', default=False, type=lambda x: bool(strtobool(x)))

    # Saving
    parser.add_argument('--savemodel', default=True, type=lambda x: bool(strtobool(x)))

    # Wrapper specific arguments
    parser.add_argument('--vec_check_nan', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--norm_rew', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--clip_rew', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--vec_normalize', default=False, type=lambda x: bool(strtobool(x)))

    # General RL Algorithm specific arguments
    parser.add_argument('--agent', type=str, default='PPO', choices=["PPO", "SAC", "DDPG", "RECPPO"])
    parser.add_argument('--agent-config', type=str, default='default')

    parser.add_argument("--num_envs", type=int, default=12,
                        help="The number of parallel environments.")
    parser.add_argument('--total_timesteps', type=str, default=gen_params['total_timesteps'],
                        help="Total number of the experiments.")
    parser.add_argument('--max_env_steps', type=int, default=gen_params['max_env_steps'],
                        help="Total timesteps of one episode.")
    parser.add_argument("--learning_rate", type=str, default=gen_params['learning_rate'],
                        help="The learning rate of the optimizer.")

    parser.add_argument('--discount', type=int, default=gen_params['discount'])
    parser.add_argument('--threshold', type=int, default=gen_params['threshold'])
    parser.add_argument('--batch_size', type=int, default=gen_params['batch_size'])
    parser.add_argument('--num_steps', type=int, default=gen_params['num_steps'])

    # PPO specific
    parser.add_argument('--clip_range', type=int, default=def_ppo_params['clip_range'])
    parser.add_argument('--ent_coef', type=int, default=def_ppo_params['ent_coef'])

    parser.add_argument('--optimizer', type=str, default='default')
    parser.add_argument('--optimizer-config', type=str, default='default')

    # Wandb
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, the experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="The entity (team) of wandb's project")
    parser.add_argument('--wandb_rootlog', type=str, default="/wandb")

    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances")

    return parser.parse_args()
