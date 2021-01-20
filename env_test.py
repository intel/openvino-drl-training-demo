import gym
import panda_hover

env = gym.make('panda_hover-v0', gui=True)

for x in range(3):
    done = False
    while not done:
        rand_action = env.action_space.sample()
        obs, reward, done, info = env.step(rand_action)
    env.reset()
    print("Resetting")
