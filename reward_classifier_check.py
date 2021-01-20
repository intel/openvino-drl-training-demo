import gym
import panda_hover

env = gym.make('panda_hover-v0')

goal_pos = [0.5, 0.25]

for x in range(10):
    env.move(goal_pos)
    img = env.get_image()
    output = env.reward_classifier(img)
    print(output)

for x in range(10):
    env.reset()
    rand_action = env.action_space.sample()
    obs, reward, done, info = env.step(rand_action)
    print(reward)
