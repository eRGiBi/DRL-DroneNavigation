
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
    parser.add_argument('--max_steps', type=str, default=5e6,
                        help="total number of the experiments")
    parser.add_argument('--max_env_steps', type=int, default=4096,
                        help="total timesteps of one episode")
    parser.add_argument("--learning_rate", type=str, default=1e-3,
                        help="the learning rate of the optimizer")

    # RL Algorithm specific arguments
    parser.add_argument('--agent', type=str, default='PPO')
    parser.add_argument('--agent-config', type=str, default='default')
    parser.add_argument('--discount', type=int, default=0.999)
    parser.add_argument('--threshold', type=int, default=0.3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_steps', type=int, default=2048)

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

    return parser.parse_args()