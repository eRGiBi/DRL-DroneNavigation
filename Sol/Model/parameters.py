"""

Default parameters for different models in dictionaries.

PPO

SAC

"""

gen_params = {
    'seed': 1,
}

rl_algorithms = {

}


def_ppo_params = {
    'clip_range': 0.2,
    'ent_coef': 0,
    'discount': 0.999,
    'threshold': 0.3,
    'batch_size': 2048,
    'num_steps': 2048
}

def_sac_params = {
    'discount': 0.99,
    'tau': 0.005,
    'policy_delay': 2,
    'target_update_interval': 1,
    'automatic_entropy_tuning': True
}
