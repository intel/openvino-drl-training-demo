import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import math
from .reward_inference import init_model, inference
from .vino_inference import init_vino_model, vino_inference
import time

class PandaHoverEnv(gym.Env):

    def __init__(self, gui=True, vino=False, device='CPU'):
        self._vino = vino
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self._orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        self._object_position = [0.65,-0.3,0.05]
        self._hover_z = 0.1
        self._horizon = 0
        self._max_episode_length = 10
        self._obs_low = np.array([0.4, -0.4])
        self._obs_high = np.array([0.7, 0.4])
        self.action_space = spaces.Box(np.ones(2)*-0.1, np.ones(2)*0.1)
        self.observation_space = spaces.Box(self._obs_low, self._obs_high)
        self._robot_hover_goal = self._object_position
        self._robot_hover_goal[2] = self._hover_z
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.55,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-10)
        urdf_root_path = pybullet_data.getDataPath()
        planeUid = p.loadURDF(os.path.join(urdf_root_path,"plane.urdf"), basePosition=[0,0,-0.65])
        self._robot = p.loadURDF(os.path.join(urdf_root_path, "franka_panda/panda.urdf"),useFixedBase=True)
        tableUid = p.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        object = p.loadURDF(os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=self._object_position)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        if self._vino:
            self._model = init_vino_model('models/squeeze_reward_classifier', device)
        else:
            self._model = init_model('squeezenet', 'models/squeeze_reward_classifier')
        self.inference_time = 0
        self.robot_moving_time = 0
        self.reset()



    def reset(self):
        self._horizon = 0
        init_joint_states = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        for i in range(7):
            p.resetJointState(self._robot,i, init_joint_states[i])
        curr_pose = self._get_robot_position()
        new_pose = curr_pose
        new_pose[2] = self._hover_z
        self._move_robot(new_pose)
        observation = self._get_robot_position()[0:2]
        return observation

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dx = action[0]
        dy = action[1]
        curr_pose = p.getLinkState(self._robot, 11)
        curr_position = curr_pose[0]
        new_position = [curr_position[0] + dx, curr_position[1] + dy, self._hover_z]
        action_taken = False
        if (self._obs_low[0] <= new_position[0] <= self._obs_high[0]) and (self._obs_low[1] <= new_position[1] <= self._obs_high[1]):
            self._move_robot(new_position)
            action_taken = True
        time_1 = time.time()
        classifier_result, classifier_result_probabilities = self._reward_classifier()
        time_2 = time.time()
        self.inference_time += (time_2 - time_1)
        observation = self._get_robot_position()[0:2]
        info = {}
        self._horizon = self._horizon + 1
        done = False
        if not classifier_result:
            done = True
            print("Goal Achieved")
            reward = 100
        else:
            if action_taken:
                reward = -1
            else:
                reward = -5
        if self._horizon == self._max_episode_length:
            done = True
        return observation, reward, done, info

    def render(self):
        print("Robot's Distance 2 Object: ", self._get_robot_2_object_distance())
        print("Robot's Current Position: ", self._get_robot_position())

    def close(self):
        p.disconnect()


    def _get_image(self):
        (_, _, rgb, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=self._view_matrix,
                                              projectionMatrix=self._proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (720,960, 4))
        rgb = rgb[:, :, :3]
        bgr = rgb[:,:, ::-1]
        return bgr


    def _move_robot(self, position):
        time_1 = time.time()
        for x in range(500):
            joint_poses = p.calculateInverseKinematics(self._robot,11,position, self._orientation)
            p.setJointMotorControlArray(self._robot, list(range(7))+[9,10], p.POSITION_CONTROL, list(joint_poses))
            p.stepSimulation()
        time_2 = time.time()
        self.robot_moving_time += time_2 - time_1

    def _get_robot_position(self):
        return np.array(p.getLinkState(self._robot, 11)[0])

    def _get_robot_2_object_distance(self):
        return np.linalg.norm(np.array(self._robot_hover_goal) - np.array(self._get_robot_position()))

    def _reward_classifier(self):
        image = self._get_image()
        if self._vino:
            classifier_result = vino_inference(self._model, image)
        else:
            classifier_result = inference(self._model, image)
        return classifier_result
