
# Copyright (C) 2020 Intel Corporation
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
# and limitations under the License.

import gym
import panda_hover
from stable_baselines3 import SAC
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ov', '--vino', action='store_true',
    help="uses openVino for classifier inference")
parser.add_argument('-g', '--gui', action='store_true',
    help="gui for robot visualization")


args = parser.parse_args()

env = gym.make('PandaHover-v0', gui=args.gui, open_vino=args.vino)

model = SAC("MlpPolicy", env, verbose=1)

start = time.time()
model.learn(total_timesteps=8000)
final = time.time()
model.save("sac_hover_agent")

print("Total Time (s): ", final-start)
print("Total Inference Time (s): ", env.inference_time)
print("Total Robot Moving Time (s): ", env.robot_moving_time)
