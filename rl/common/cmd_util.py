# Inspired from OpenAI Baselines
import gym
from rl.common.vec_env.subproc_env_vec import SubprocVecEnv
from rl.common import set_global_seeds


def make_gym_env(env_id, num_env=2, seed=123, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, SubprocVecEnv for Gym Environments.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk

    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
