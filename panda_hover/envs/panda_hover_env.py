# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.from .openvino_model import OpenVinoModel

from .pytorch_model import PyTorchModel
from gym import spaces
import numpy as np
import pybullet_data
import pybullet
import math
import time
import gym
import os

class PandaHoverEnv(gym.Env):

    _ROBOT_MIN_X = 0.4
    _ROBOT_MAX_X = 0.7
    _ROBOT_MIN_Y = -0.4
    _ROBOT_MAX_Y = 0.4
    _MIN_ACTION = -0.1
    _MAX_ACTION = 0.1
    _HOVER_Z = 0.1
    _OBJ_POS = [0.65,-0.3,0.05]
    _GRIPPER_ORIENT = pybullet.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
    _MAX_EPISODE_LENGTH = 10

    def __init__(self, gui=True, open_vino=False, device='CPU'):
        self._horizon = 0
        self._robot_hover_goal = self._OBJ_POS
        self._robot_hover_goal[2] = self._HOVER_Z
        self._open_vino = open_vino
        # Define observation and action space info for as per OpenAI gym.ENV convention
        self._obs_low = np.array([self._ROBOT_MIN_X, self._ROBOT_MIN_Y])
        self._obs_high = np.array([self._ROBOT_MAX_X, self._ROBOT_MAX_Y])
        self.action_space = spaces.Box(np.ones(2)*self._MIN_ACTION, np.ones(2)*self._MAX_ACTION)
        self.observation_space = spaces.Box(self._obs_low, self._obs_high)
        # Initialize PyBullet Simulation and load Plane, Robot, Table, Object, & Camera
        if gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self._view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.55,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        self._proj_matrix = pybullet.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        pybullet.resetSimulation()
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        pybullet.setGravity(0,0,-9.8)
        urdf_root_path = pybullet_data.getDataPath()
        plane = pybullet.loadURDF(os.path.join(urdf_root_path,"plane.urdf"), basePosition=[0,0,-0.65])
        self._robot = pybullet.loadURDF(os.path.join(urdf_root_path, "franka_panda/panda.urdf"),useFixedBase=True)
        table = pybullet.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        object = pybullet.loadURDF(os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=self._OBJ_POS)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
        #Choose between OpenVino or PyTorch for Inference Engine
        if self._open_vino:
            self._model = OpenVinoModel('models/squeeze_reward_classifier', device)
        else:
            self._model = PyTorchModel('models/squeeze_reward_classifier')
        self.inference_time = 0
        self.robot_moving_time = 0
        self.reset()

    def reset(self):
        self._horizon = 0
        # Resets robot to home position
        init_joint_states = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        for i in range(7):
            pybullet.resetJointState(self._robot,i, init_joint_states[i])
        curr_pose = self._get_robot_position()
        new_pose = curr_pose
        new_pose[2] = self._HOVER_Z
        self._move_robot(new_pose)
        observation = self._get_robot_position()[0:2]
        return observation

    def step(self, action):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        dx = action[0]
        dy = action[1]
        curr_pose = pybullet.getLinkState(self._robot, 11)
        curr_position = curr_pose[0]
        new_position = [curr_position[0] + dx, curr_position[1] + dy, self._HOVER_Z]
        action_taken = False
        # check if the action keeps the robot within its workspace
        if (self._obs_low[0] <= new_position[0] <= self._obs_high[0]) and (self._obs_low[1] <= new_position[1] <= self._obs_high[1]):
            self._move_robot(new_position)
            action_taken = True
        classifier_result = self._reward_classifier()
        observation = self._get_robot_position()[0:2]
        info = {}
        self._horizon = self._horizon + 1
        done = False
        # if the Reward Classifier Neural Network returns 0, robot has achieved the task
        if not classifier_result:
            done = True
            print("Goal Achieved")
            reward = 100
        else:
            if action_taken:
                reward = -1
            else:
                reward = -5 # robot tried to move outside its workspace. Penalize this
        if self._horizon == self._MAX_EPISODE_LENGTH:
            done = True
        return observation, reward, done, info

    def render(self):
        print("Robot's Distance 2 Object: ", self._get_robot_2_object_distance())
        print("Robot's Current Position: ", self._get_robot_position())

    def close(self):
        pybullet.disconnect()

    def _get_image(self):
        (_, _, rgb, _, _) = pybullet.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=self._view_matrix,
                                              projectionMatrix=self._proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (720,960, 4))
        rgb = rgb[:, :, :3]
        bgr = rgb[:,:, ::-1]
        return bgr

    def _move_robot(self, position):
        time_1 = time.monotonic()
        joint_poses = pybullet.calculateInverseKinematics(self._robot,11,position, self._GRIPPER_ORIENT)
        pybullet.setJointMotorControlArray(self._robot, list(range(7))+[9,10], pybullet.POSITION_CONTROL, list(joint_poses))
        for x in range(500):
            pybullet.stepSimulation()
        time_2 = time.monotonic()
        self.robot_moving_time += time_2 - time_1

    def _get_robot_position(self):
        return np.array(pybullet.getLinkState(self._robot, 11)[0])

    def _get_robot_2_object_distance(self):
        return np.linalg.norm(np.array(self._robot_hover_goal) - np.array(self._get_robot_position()))

    def _reward_classifier(self):
        time_1 = time.monotonic()
        image = self._get_image()
        classifier_result = self._model.inference(image)
        time_2 = time.monotonic()
        self.inference_time += (time_2 - time_1)
        return classifier_result
