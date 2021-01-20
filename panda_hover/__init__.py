from gym.envs.registration import register

register(
    id='panda_hover-v0',
    entry_point='panda_hover.envs:PandaHoverEnv',
)
