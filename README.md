# Deep Reinforcement Learning Simulator and Algorithms for optimal pathfinding

### Working model based on SB3 and pybullet-drones with strictly positional observations

<p align="center">
  <img src="assets/gifs/example-speed.gif" alt=""/>
</p>

---

### 1. Introduction

The autonomous navigation of drones is a challenging problem that is yet to be solved.
It requires the development of efficient algorithms for planning,
control and intelligent decision-making amidst unexpected observations. 
Deep reinforcement learning (DRL) has shown promise in solving these problems,
but it is still an open question how to best apply it.

My original goal was to develop a DRL-based framework for autonomous drone racing, 
but this same model could be utilized for other, more relevant real-world scenarios, 
such as search and rescue, surveillance and package delivery.

The simulator is designed to be lightweight, flexible and modular,
based on the PyBullet-drones environment, 
with the algorithms mainly borrowed from the Stable-Baselines3 library.
I also have used Tensorflow's TF-Agents, the OpenAI Gym library,
and run tests using AirSim, CleanRL, and the RLLib library.

The focus is on the Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) algorithms, 
which are optimized for continuous control tasks.

I evaluated the performance of the algorithms with a variety of differing hyperparameters 
and visualized the results in different ways. 

I also implemented and tested a variety of novel and research-based reward functions.

---

### 2. Requirements
To install dependencies in a Conda, miniconda or Poetry environment:

```
$ pip install -r requirements.txt
```

Adding path might help:    $env:PYTHONPATH = "...\RL"



To install PyTorch on Windows or Linux with CUDA 11.8 for GPU support:
```
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
To install TensorFlow with GPU support (not need for main algorithm):
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"

# Verify the installation:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The CleanRL and RLLib implementations also need to be installed separately.

---

### 3. Basic operation

First, the simulation_controller reads in the args, initializes the targets (the Waypoints class), 
then starts the chosen function of the PBDroneSimulator class that controls
the whole learning or testing process.

---

### 4. Example


Accessing tensorboard logs:
```
tensorboard --logdir ./Sol/logs/
```

Example args for a full training process on Windows (using gui significantly slows learning):

```
$ python Sol/Model/simulation_controller.py --agent PPO --run_type full  --wandb f --savemodel t --gui f --norm_rew f --lib sb3 --num_envs 12 
``` 
For Ubuntu:
```
python3 ./Sol/Model/simulation_controller.py --agent PPO --run_type full --wandb f
``` 

For Ubuntu with WSL:
```
 python3 Sol/Model/simulation_controller.py --agent PPO --run_type full --wandb False --savemodel False
 ```
On root installation for activating the example conda environment:
```
 source activate  /root/miniconda3/envs/RL-WSL
```


### Success models

Loads and tests a preselected model. 
All models are saved in the model_chkpts folder. 
I sampled a few good policies in the function.

Might need to tweak the observation (and action) spaces prior to launching.


```
$  python Sol/Model/simulation_controller.py --agent PPO --run_type saved  --wandb f  
``` 


### Utilities
I implemented a few functions for tf.events manipulation which visualizes the training process according 
to selected metrics.
   
    python Sol/Utilities/TensorBoardManager.py   

Trajectory visualizations from collected data during a full training process.
(requires collecting a vast number of episodes).

    python Sol/Utilities/TrajectoryVisualizer.py  

Value function regression with other methods (requires collecting rollouts):

    python Sol/Model/Policies/alt_methods.py

---

### 5. Results

I optimized the learning environment and fine-tuned the PPO hyperparameters.
It turned out to be an effective solution for the drone navigation problem, but with a few assumptions.
The PPO agent learns to navigate through the waypoints in about 4 hours of training on a low-spec machine.


![](assets/comb.png)

With not perfectly fine-tuned hyperparameters, SAC manages to learn a circle track in 15 hours of training. 

---

### 6. Notes

The OpenGl 3 engine does not work in virtual machines, so in order to have a visual representation of the simulation,
it is necessary to run the simulation with OpenGL 2, set in the modified BaseAviary class, as such: 
```
p.connect(p.GUI, options="--opengl2")
```
Further: only OpenGL3 works in Windows. Using Ubuntu Virtual Machine this stackoverflow article might be useful: 
https://askubuntu.com/questions/1352158/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04

On Ubuntu, with an NVIDIA card, if encountered a "Failed to create and OpenGL context" message, launch nvidia-settings 
with "Performance Mode PRIME Profile," reboot and try again.

StableBaselines3 and other packages that have '_' in them might not be installed or recognized correctly.

---

### 7. To Do

1. Frame skipping.
2. Added noise test to ease the sim-to-real transfer.

