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
# and limitations under the License.

from stable_baselines3 import SAC
import panda_hover
import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-ov', '--vino', action='store_true',
    help="uses openVino for classifier inference")
parser.add_argument('-g', '--gui', action='store_true',
    help="gui for robot visualization")


args = parser.parse_args()

env = gym.make('PandaHover-v0', gui=args.gui, open_vino=args.vino)
model = SAC.load("sac_hover_agent")

print("Starting Inference")
success = 0
num_inferences = 100
for x in range(0, num_inferences):
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done,_ = env.step(action)

    if reward == 100:
        success = success + 1

print("Succes Rate %: ", (100*success)/num_inferences)
