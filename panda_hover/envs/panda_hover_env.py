import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from numpy import linalg
from .ft_inf import initialize_model
from .ft_inf import inference
import os
class PandaHoverEnv(gym.Env):

    robot = None
    viewMatrix = None
    projectionMatrix = None
    z = 0
    episode_limit = 15
    bb_neg = np.array([0.3, 0.15])
    bb_pos = np.array([0.7, 0.7])
    ball_pos = [0.5, 0.25, 0.1]
    horizon = 0


    def __init__(self, gui=False):

        directory = os.getcwd() + "/panda_hover/envs/"

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(directory)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()

        plane = p.loadURDF("plane.urdf")
        print(directory)

        #self.robot = p.loadURDF('/home/intel/bullet_vino/panda_hover/envs/panda/panda.urdf', [0, 0.5, 0], useFixedBase=1)
        self.robot = p.loadURDF(directory + 'panda/panda.urdf', [0, 0.5, 0], useFixedBase=1)

        #ball = p.loadURDF('/home/intel/bullet_vino/panda_hover/envs/panda/ball.urdf')
        ball = p.loadURDF(directory + 'panda/ball.urdf')
        p.changeVisualShape(ball, -1, rgbaColor=[1,0,0,1])

        p.changeDynamics(ball,-1,restitution=.95, linearDamping = 1e-2, angularDamping = 1e-2)

        default_ball_ori = [0,0,0,1]
        p.resetBasePositionAndOrientation(ball, self.ball_pos, default_ball_ori)

        p.setGravity(0, 0, -9.81)

        p.setRealTimeSimulation(0)

        self.init_joints = [-6.030826336709875e-12, 0.4590974981626072, 7.786621936742484e-12, -2.2531715942788773, -9.705404415856309e-12, 2.7122379648773216, 1.3945653759819635e-11]

        self.viewMatrix = p.computeViewMatrix(cameraEyePosition=[1, 0.5, 1.5], cameraTargetPosition=[0.6, 0.5,0.1], cameraUpVector=[0, 0, 1])

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)


        for x in range(100):
            p.setJointMotorControlArray(
                self.robot, range(7), p.POSITION_CONTROL,
                targetPositions=self.init_joints)
            p.stepSimulation()
            joint_positions = [j[0] for j in p.getJointStates(self.robot, range(7))]
            state = p.getLinkState(self.robot, 7)

        self.z = state[0][2]

        action_low = np.ones(2) * -0.1
        action_high = np.ones(2) * 0.1

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.bb_neg, high=self.bb_pos, dtype=np.float32)

        self.model = initialize_model('resnet', directory + 'single_goal_classifier_resnet')


    def move(self, position):

        pose = [position[0], position[1], self.z]
        target_j = p.calculateInverseKinematics(self.robot, 7, pose, [1,0,0,0])
        for x in range(500):
            p.setJointMotorControlArray(self.robot, range(7), p.POSITION_CONTROL, targetPositions=target_j)
            p.stepSimulation()

    def get_position(self):
        state = p.getLinkState(self.robot, 7)
        return np.array(state[0][0:2])

    def get_image(self):
        width, height, rgb, depth, seg = p.getCameraImage(224, 224, self.viewMatrix, self.projectionMatrix)
        rgb = rgb[:, :, 0:3]
        bgr = rgb[:,:, ::-1]
        return bgr

    def distance_2_ball(self, pos):
        dist = self.ball_pos[0:2] - pos
        return linalg.norm(dist)

    def reward_classifier(self, img):
        return inference(img, self.model)

    def step(self, action):

        curr_pose = self.get_position()
        new_pos = curr_pose + action
        taken = False
        #print("Command: ", new_pos)

        if all(self.bb_neg <= new_pos) and all(new_pos <= self.bb_pos):
             self.move(new_pos)
             taken = True
        #else:
            #print("Action not taken")


        #print("Actual: ", self.get_position())

        img = self.get_image()

        pos = self.get_position()

        done = False
        nn_output = self.reward_classifier(img)

        if taken:
            reward = nn_output
        else:
            reward = -300

        if nn_output > -2:

            reward = 2000
            done = True
            dist = self.distance_2_ball(pos)
            print("Goal Achieved: ", dist)
            print(pos)

        elif self.horizon >= self.episode_limit - 1:
            done = True
            reward = -500
        self.horizon = self.horizon + 1
        observation = pos
        info = {}
        return observation, reward, done, info

    def reset(self):
        for x in range(100):
            p.setJointMotorControlArray(self.robot, range(7), p.POSITION_CONTROL, targetPositions=self.init_joints)
            p.stepSimulation()
        observation = self.get_position()
        self.horizon = 0
        return observation
