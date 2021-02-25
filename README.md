# openVino_Blog

This code repo showcases how one could use Intel's OpenVino to accelerate Deep Reinforcement Learning Training. 
Specifically, this is for RL problems which leverage pre-trained goal classifiers for their reward function. The same 
idea can be applied for RL problems which leverage pre-trained autoencoders for state-space reduction. 

## Reproducing this Repo

### Installing Pre-Requisite Software

Note: The code in this repo was validated on python3.6. Highly reccomended to create a seperate python3.6 virtual environment for this code!

Step 1: Check to see if the correct version of openMPI is installed on your system (should be 2.1.1)

```console
mpirun --version
```

Step 2: install stable baselines dependencies

```console
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Step 3: Git clone this library 

Step 4: Install all the python packages (reccomended to this in a python virtual environment)
```console
pip3 install -r requirements.txt
```
### Installing and Veryifying OpenVino Install
Step 5: Install Intel's OpenVino Tool https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html

Step 6: After intstalling openvino, enable it by running 

```console
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Step 7: Check to see what are the availble devices for openVino to use
```console
python /opt/intel/openvino_2021/inference_engine/samples/python/hello_query_device/hello_query_device.py
```
### Optimzing the Reward Network on your machine
Step 8: Create openVino binares of the pre-trained reward classifier 
```console
python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model panda_hover/envs/single_goal_classifier_resnet.onnx
```

Step 9: Move generated binaries to proper directory

```console
mv single_goal_classifier_resnet.* panda_hover/envs/
```
### Training the Agent
Step 10: Run vino_agent_training.py. Note the total exection time + training time outputed at the end

Step 11: Run vino_agent_training.py after enabiling vino=True in line 7. Compare the total execution time and training time to Step 10 (should be lower)

Step 12: Run inference.py to see the trained robot!


## Environment Paramters

```python
env = gym.make('panda_hover-v0', gui=False, vino=False, device='CPU')
```

When intialzing the environment, there are 3 important parameters when initializing the environment

1.gui: whether to run the simulation software with a gui or headless

2.vino: whether to use openVino as the inference engine for the reward classifier network or pytorch

3.device: which device to use as the inference engine (CPU, GPU, MYRIAD). This is only applicable if vino parameter is true

## Enviornment Info 

![Screenshot](4_n.png)

The goal of our gym environment is for our robot to learn to navigate to the postion of the red spehere using vision. 

The state is the robot' end effector's x,y position.

The action is the robot's dx, dy postion

The reward function is output of the goal classifier network normalized to probabilties from -100 to 0. The Input to the network is an image from the camera
