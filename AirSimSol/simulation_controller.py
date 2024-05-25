""""
    Abandoned, for AirSim has huge computing requirements.
"""

import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


def main():
    env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "airgym:airsim-drone-sample-v0",
                    ip_address="127.0.0.1",
                    step_length=0.25,
                    image_shape=(84, 84, 1),
                )
            )
        ]
    )

    env = VecTransposeImage(env)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=0.00025,
        verbose=2,
        batch_size=512,
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log=None,
    )

    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path=".",
        log_path=".",
        eval_freq=10000,
    )
    callbacks.append(eval_callback)

    model.learn(
        total_timesteps=int(5e5),
        tb_log_name="airsim_drone_run_" + str(time.time()),
        callback=callbacks
    )

    # Save policy weights
    model.save("airsim_drone_policy")


if __name__ == '__main__':
    main()
