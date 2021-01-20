import gym
import panda_hover
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
import time

env = gym.make('panda_hover-v0', gui=True)
model = PPO1.load("ppo_hover_agent", verbose=0)
model.set_env(env)

print("Starting Inference")
for x in range(0, 25):
    is_done = False
    obs = env.reset()
    print(obs)
    steps = 0
    while not is_done:
        action, _ = model.predict(obs)
        obs, reward, is_done,_ = env.step(action)
        #env.render()

        steps = steps + 1
    #print(env._q)
    if steps > 15:
        print("Goal Not Reached")
