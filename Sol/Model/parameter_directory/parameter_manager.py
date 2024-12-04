"""

    Default parameters for different models in dictionaries.
    PPO, SAC

     # TODO

"""

import yaml

def load_default_params():
    """Load default parameters for different models in dictionaries."""

    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)

    return params

gen_params = {
    'seed': 1,
    'num_envs': 12,
    'learning_rate': 3e-4,
    'total_timesteps': 10e6,
    'max_env_steps': 4096,

    'discount': 0.99,
    'threshold': 0.3,
    'batch_size': 128,
    'num_steps': 2048
}

def_ppo_params = {
    'clip_range': 0.1,
    'ent_coef': 0.2,
}

best_ppo_params = {
    'clip_range': 0.2,
    'ent_coef': 0.01,
}

def_sac_params = {
    'discount': 0.99,
    'tau': 0.005,
    'policy_delay': 2,
    'target_update_interval': 1,
    'automatic_entropy_tuning': True
}
