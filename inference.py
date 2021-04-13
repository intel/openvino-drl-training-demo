import gym
import panda_hover
from stable_baselines3 import SAC

env = gym.make('panda_hover-v0', gui=False)
model = SAC.load("sac_hover_agent")

print("Starting Inference")
success = 0
num_inferences = 25
for x in range(0, 25):
    is_done = False
    obs = env.reset()
    steps = 0
    while not is_done:
        action, _ = model.predict(obs)
        obs, reward, is_done,_ = env.step(action)
        #env.render()
        steps = steps + 1

    if reward == 2000:
        success = success + 1

print("Succes Rate %: ", (100*success)/num_inferences)
