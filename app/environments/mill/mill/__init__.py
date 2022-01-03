from gym.envs.registration import register

register(
    id='Mill-v0',
    entry_point='mill.envs.mill:MillEnv',
)
