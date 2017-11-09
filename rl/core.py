# -*- coding: utf-8 -*-
import time
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
import util


class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """
    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, run_env_parally_in_n_processes=1, environment_arguments=[]):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
                IF run_env_parally_in_n_processes is set ot anything other than 1 then a call to env is
                assumed to return an object of Environment that the agent interacts with.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same actions without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single actions only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observations: actions`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random actions is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        # visualization can not happen as before when there are multiple envs. Revert to
        # single environment mode.
        if run_env_parally_in_n_processes == 1:
            callbacks._set_env(env)
            envs = [env]
        else:
            envs = []
            # create lots and lots of instances of environments in different processes
            for i in range(run_env_parally_in_n_processes):
                envs.append(util.EnvironmentInstance(env, environment_arguments))

        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = [0]*run_env_parally_in_n_processes
        self.step = 0
        observations = []
        did_abort = False
        try:
            while self.step < nb_steps:
                if not observations:  # start of a new episode
                    for i in range(run_env_parally_in_n_processes):
                        callbacks.on_episode_begin(episode[i])
                    episode_step = [0]*run_env_parally_in_n_processes
                    episode_reward = [0]*run_env_parally_in_n_processes

                    # Obtain the initial observations by resetting the environment.
                    self.reset_states()
                    for i in range(run_env_parally_in_n_processes):
                        observations.append(deepcopy(envs[i].reset()))
                    if self.processor is not None:
                        for i in range(run_env_parally_in_n_processes):
                            observations[i] = self.processor.process_observation(observations[i])
                    assert observations is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)

                    actions = [None]*run_env_parally_in_n_processes
                    done = [None]*run_env_parally_in_n_processes
                    rewards = [None]*run_env_parally_in_n_processes
                    info = [None]*run_env_parally_in_n_processes

                    for i in range(run_env_parally_in_n_processes):
                        for _ in range(nb_random_start_steps):
                            if start_step_policy is None:
                                actions[i] = envs[i].action_space.sample()
                            else:
                                actions[i] = start_step_policy(observations[i])

                            if self.processor is not None:
                                actions[i] = self.processor.process_action(actions[i])

                            callbacks.on_action_begin(actions[i])
                            observations[i], rewards[i], done[i], info[i] = envs[i].step(actions[i])

                            observations = deepcopy(observations)
                            if self.processor is not None:
                                    observations[i], rewards[i], done[i], info[i] = self.processor.process_step(observations[i], rewards[i], done[i], info[i])
                            callbacks.on_action_end(actions)
                            if done[i]:
                                warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                                observations = deepcopy(envs[i].reset())
                                if self.processor is not None:
                                    observations = self.processor.process_observation(observations)
                                break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observations is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the actions
                # (forward step) and then use the rewards to improve (backward step).
                for i in range(run_env_parally_in_n_processes):
                    actions[i] = self.forward(observations[i], i)

                    if self.processor is not None:
                        actions = self.processor.process_action(actions)
                    rewards[i] = 0.
                    done[i] = False

                accumulated_info = [{}]*run_env_parally_in_n_processes

                r = [None]*run_env_parally_in_n_processes
                for _ in range(action_repetition):
                    for i in range(run_env_parally_in_n_processes):
                        if done[i]:
                            continue
                        callbacks.on_action_begin(actions[i])
                        observations[i], r[i], done[i], info[i] = envs[i].step(actions[i])
                        observations = deepcopy(observations)
                        if self.processor is not None:
                            observations[i], r[i], done[i], info[i] = self.processor.process_step(observations[i], r[i], done, info[i])
                        for key, value in info[i].items():
                            if not np.isreal(value):
                                continue
                            if key not in accumulated_info[i]:
                                accumulated_info[i][key] = np.zeros_like(value)
                            accumulated_info[i][key] += value
                        callbacks.on_action_end(actions)
                        rewards[i] += r[i]

                for i in range(run_env_parally_in_n_processes):
                    if nb_max_episode_steps and episode_step[i] >= nb_max_episode_steps - 1:
                        # Force a terminal state.
                        done = [True]*run_env_parally_in_n_processes
                    metrics = self.backward(rewards[i], i, terminal=done[i])
                    episode_reward[i] += rewards[i]

                    step_logs = {
                        'actions': actions[i],
                        'observations': observations[i],
                        'reward': rewards[i],
                        'metrics': metrics,
                        'episode': episode[i],
                        'info': accumulated_info[i],
                    }
                    callbacks.on_step_end(episode_step[i], step_logs)
                    episode_step[i] += 1
                    self.step += 1

                    if done[i]:
                        # We are in a terminal state but the agent hasn't yet seen it. We therefore
                        # perform one more forward-backward call and simply ignore the actions before
                        # resetting the environment. We need to pass in `terminal=False` here since
                        # the *next* state, that is the state of the newly reset environment, is
                        # always non-terminal by convention.
                        self.forward(observations[i], i)
                        self.backward(0., i, terminal=False)

                        # This episode is finished, report and reset.
                        episode_logs = {
                            'episode_reward': episode_reward[i],
                            'nb_episode_steps': episode_step[i],
                            'nb_steps': self.step,
                        }
                        callbacks.on_episode_end(episode[i], episode_logs)

                        episode[i] += 1
                        observations = []
                        episode_step = [0]*run_env_parally_in_n_processes
                        episode_reward = [0]*run_env_parally_in_n_processes
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins."
        """
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py


class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) 
        
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()
