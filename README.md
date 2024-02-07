
# Deep reinforcement learning Simulator and Algorithms for optimal path finding in autonomous drone navigation

### Based on Stable-Baseline3 and PyBullet-drones, optimized for PPO and SAC


### 1. Introduction

The autonomous navigation of drones is a challenging problem that is yet to be solved.
It requires the development of efficient algorithms for path planning,
control, and intelligent decision-making when unexpected events occur. 
Deep reinforcement learning (DRL) has shown promise in solving these problems,
but it is still an open question how to best apply it to the domain of autonomous drone navigation.

My original goal with this project was to develop a DRL-based framework for autonomous drone racing, 
but this same model could be utilized for other, perhaps more relevant real-world scenarios, such as search and rescue, surveillance, and package delivery.

The simulator is based on the PyBullet-drones environment, and the algorithms are based on the Stable-Baselines3 library. 
I also have used Tensorflow's TF-Agents library, the OpenAI Gym library, and run tests using AirSim.
I focus on the Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) algorithms, which are optimized for continuous control tasks. 
I evaluate the performance of the algorithms with a variety of differing hyperparameters.

### 2. Requirements
To install dependencies in a Conda or Poetry environment:

```
$ pip install -r requirements.txt
```

### 3. Example:
Example args:

```
$ your python RL\Sol\Model\simulation_controller.py --agent PPO --run_type full --wandb
 False --savemodel False --max_steps 10e6
```
