import gym
import panda_hover
from stable_baselines3 import SAC
import time

env = gym.make('PandaHover-v0', gui=False, vino=True)

model = SAC("MlpPolicy", env, verbose=1)

start = time.time()
model.learn(total_timesteps=6000)
final = time.time()
model.save("sac_hover_agent")

print("Total Time (s): ", final-start)
print("Total Inference Time (s): ", env.inference_time)
print("Total Robot Moving Time (s): ", env.robot_moving_time)
