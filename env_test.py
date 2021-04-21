import gym
import panda_hover


env = gym.make('PandaHover-v0', gui=True, vino=True)

for x in range(5):
    observation, reward, done, info = env.step([0.219, -0.301])


for x in range(1):
    observation = env.reset()
    done = False
    while not done:
        rand_action = env.action_space.sample()
        observation, reward, done, info = env.step(rand_action)
        env.render()
