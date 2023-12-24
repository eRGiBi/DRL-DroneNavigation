import gymnasium
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecCheckNan, VecNormalize, DummyVecEnv
import numpy as np
from stable_baselines3 import PPO
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv


# env = gymnasium.make("PyFlyt/QuadX-Waypoints-v1", render_mode="human", num_targets=1)
#
# term, trunc = False, False
# obs, _ = env.reset()
# while not (term or trunc):
#     obs, rew, term, trunc, _ = env.step(env.action_space.sample())


env = gymnasium.make("PyFlyt/QuadX-Waypoints-v1", render_mode="human", num_targets=1)

env = make_vec_env("PyFlyt/QuadX-Waypoints-v1", n_envs=16, seed=0, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(render_mode="human", num_targets=1))

env = FlattenWaypointEnv(env, context_length=1)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)

model.save("ppo_quadx_waypoints")

# Load the trained model
loaded_model = PPO.load("ppo_quadx_waypoints")

obs = env.reset()
while True:
    action, _ = loaded_model.predict(np.array(obs))
    obs, reward, done, trunc = env.step(action)
    env.render()
    print(obs, reward, done, trunc)
    if done:
        obs = env.reset()



