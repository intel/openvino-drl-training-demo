from gym.envs.registration import register

register(
    id='PandaHover-v0',
    entry_point='panda_hover.envs:PandaHoverEnv',
)
