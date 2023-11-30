import time

import numpy as np
from stable_baselines3 import PPO

# from PyBullet import BaseAviary
from PyBullet.enums import Physics
from Sol.DroneEnvironment import DroneEnvironment

# from tf_agents.environments import py_environment

start = time.perf_counter()

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

discount = 0.999

# env = e.MinitaurBulletEnv(render=True)


# def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB,
#         record_video=DEFAULT_RECORD_VIDEO):
#     env = DroneEnvironment()
#
#     model = ()
#
#     #### Show (and record a video of) the model's performance ##
#     env = FlyThruGateAviary(gui=gui,
#                             record=record_video
#                             )
#     logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
#                     num_drones=1,
#                     output_folder=output_folder,
#                     colab=colab
#                     )
#
#     obs, info = env.reset(seed=42, options={})
#     start = time.time()
#
#     for i in range(3 * env.CTRL_FREQ):
#
#         action, _states = model.predict(obs, deterministic=True)
#
#         obs, reward, terminated, truncated, info = env.step(action)
#
#         logger.log(drone=0,
#                    timestamp=i / env.CTRL_FREQ,
#                    state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
#                    control=np.zeros(12)
#                    )
#         env.render()
#         print(terminated)
#         sync(i, start, env.CTRL_TIMESTEP)
#         if terminated:
#             obs = env.reset(seed=42, options={})
#     env.close()
#
#     if plot:
#         logger.plot()


# Connect to the PyBullet physics server
# physicsClient = p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)
# p.setRealTimeSimulation(0)
# Load the drone model
# drone = p.loadURDF("cf2x.urdf", [0, 0, 0])


targets = pts = [np.array([0.0, 0.0, 3.0]),
                 np.array([8.59735, -3.3286, -6.07256]),
                 np.array([1.5974, -5.0786, -4.32256]),
                 np.array([3.2474, 3.32137, -2.5725]),
                 np.array([1.3474, 1.6714, -2.07256]), ]

drone_environment = DroneEnvironment(race_track=None,
                                     target_points=targets,
                                     threshold=1,
                                     discount=1,
                                     drone=None,
                                     physics=Physics.PYB)
print("----------------------------")
print(drone_environment.action_space)
print(drone_environment.action_spec())
print(drone_environment.getDroneIds())
print(drone_environment.observation_spec())
print("----------------------------")

# tf_env = tf_py_environment.TFPyEnvironment(drone_environment)
#
# print('action_spec:', tf_env.action_spec())
# print('time_step_spec.observation:', tf_env.time_step_spec().observation)
# print('time_step_spec.step_type:', tf_env.time_step_spec().step_type)
# print('time_step_spec.discount:', tf_env.time_step_spec().discount)
# print('time_step_spec.reward:', tf_env.time_step_spec().reward)


action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

model = PPO(
    "MlpPolicy",
    DroneEnvironment(drone=None,
                     race_track=None,
                     target_points=targets,
                     threshold=1,
                     discount=discount,
                     gui=False
                     ),
    verbose=1
)
model.learn(total_timesteps=100000)

rewards = []

obs, info = drone_environment.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = drone_environment.step(action)
    rewards.append(reward)
    if terminated or truncated:
        obs, info = drone_environment.reset()
        break
print(rewards)
print(sum(rewards))

end = time.perf_counter()
print(end - start)

# # time_step = drone_environment.reset()
# time_step = drone_environment.step(action)
# # drone_environment.CLIENT.height = 100
# # drone_environment.CLIENT.height = 100
# print(time_step)
# # for _ in range(100):
#
# while time_step[1] != 10:
#     time_step = drone_environment.step(action)
#
#     print(time_step)
#
#     # if p.isConnected():
#     #     p.stepSimulation()
#
#     time.sleep(1. / 240.)  # Control the simulation speed

# Disconnect from the physics server
# p.disconnect()

# if __name__ == "__main__":
#     #### Define and parse (optional) arguments for the script ##
#     parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
#     parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
#                         metavar='')
#     parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
#                         help='Whether to record a video (default: False)', metavar='')
#     parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
#                         help='Folder where to save logs (default: "results")', metavar='')
#     parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
#                         help='Whether example is being run by a notebook (default: "False")', metavar='')
#     ARGS = parser.parse_args()
#
#     run(**vars(ARGS))
