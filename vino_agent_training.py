import gym
import panda_hover
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
import time

env = gym.make('panda_hover-v0', gui=False, vino=True, device='GPU' )

model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_log/")
start = time.time()
model.learn(total_timesteps=100000)
final = time.time()
model.save("ppo_hover_agent")

print("Total Time (s): ", final-start)
print("Model Load Time (s): ", env.model_load_time)
print("Inference Time (s): ", env.inf_time)
