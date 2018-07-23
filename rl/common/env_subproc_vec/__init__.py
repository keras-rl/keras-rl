# This code is inspired from OpenAI baseline's implementation of vec_env

from abc import ABC, abstractmethod

# TODO : Check the arguments of AbstractClass
# TODO : Add Render

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)        

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = self.action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, action):
        """
        Tell the parallel environment to take a step.
        Call step_wait() to get the results of the step.

        Do not try to call this step if the previous call
        pending.
        """

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async()

        This method returns (obs, rews, dones, infos)
        - obs : array of observations
        - rew : array of rewards
        - dones : array of episode done
        - infos : array of info 
        """
        pass

    @abstractmethod
    def close(self):
        """
        clean up environment's resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    @property 
    def unwrapped(self):
        if isinstance(self, AbstractEnvVecWrapper):
            return self.venv.wrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        AbstractEnvVec.__init__(self,
            self.num_envs = venv.num_envs,
            self.observation_space = observation_space or venv.observation_space,
            self.action_space = action_space or venv.action_space
            )

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
