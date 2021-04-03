import gym
import panda_hover
from stable_baselines3 import SAC
import time

env = gym.make('panda_hover-v0', gui=False, vino=False, device='CPU' )
#env = make_vec_env(env, n_envs=1)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_log/")

start = time.time()
model.learn(total_timesteps=10000)
final = time.time()
model.save("sac_hover_agent")

print("Total Time (s): ", final-start)
print("Model Load Time (s): ", env.model_load_time)
print("Inference Time (s): ", env.inf_time)
