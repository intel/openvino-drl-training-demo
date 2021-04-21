import gym
import panda_hover
from stable_baselines3 import SAC

env = gym.make('PandaHover-v0', gui=True, vino=True)
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
