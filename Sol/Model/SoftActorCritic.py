import numpy as np
# import reverb
import tempfile
import PIL.Image

from tf_agents.environments import suite_pybullet

from Sol.Model.Environments.PBDroneEnv import PBDroneEnv
from Sol.PyBullet.enums import Physics


class SoftAcorCritic():

    def __init__(self):
        pass

    def set_up_training(self, env):
        tempdir = tempfile.gettempdir()

        num_iterations = 1e6  # @param {type:"integer"}

        initial_collect_steps = 10000  # @param {type:"integer"}
        collect_steps_per_iteration = 1  # @param {type:"integer"}
        replay_buffer_capacity = 10000  # @param {type:"integer"}

        batch_size = 256  # @param {type:"integer"}

        critic_learning_rate = 3e-4  # @param {type:"number"}
        actor_learning_rate = 3e-4  # @param {type:"number"}
        alpha_learning_rate = 3e-4  # @param {type:"number"}
        target_update_tau = 0.005  # @param {type:"number"}
        target_update_period = 1  # @param {type:"number"}
        gamma = 0.99  # @param {type:"number"}
        reward_scale_factor = 1.0  # @param {type:"number"}

        actor_fc_layer_params = (256, 256)
        critic_joint_fc_layer_params = (256, 256)

        log_interval = 5000  # @param {type:"integer"}

        num_eval_episodes = 20  # @param {type:"integer"}
        eval_interval = 10000  # @param {type:"integer"}

        policy_save_interval = 5000  # @param {type:"integer"}

        # env = tf_py_environment.TFPyEnvironment(env)
        env = suite_pybullet.load(env)
        env.reset()
        PIL.Image.fromarray(env.render())

        print('Observation Spec:')
        print(env.time_step_spec().observation)
        print('Action Spec:')
        print(env.action_spec())


if __name__ == '__main__':
    drone_environment = PBDroneEnv(target_points=[np.array([0.0, 0.0, 1.0]),
                                                  np.array([8.59735, -3.3286, -6.07256]),
                                                  np.array([1.5974, -5.0786, -4.32256]),
                                                  np.array([3.2474, 3.32137, -2.5725]),
                                                  np.array([1.3474, 1.6714, -2.07256]), ],
                                   threshold=1,
                                   discount=1,
                                   physics=Physics.PYB,
                                   gui=True,
                                   initial_xyzs=np.array([[0, 0, 0]])
                                   )
    sac = SoftAcorCritic()
    sac.set_up_training(drone_environment)
