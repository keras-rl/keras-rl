import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList


class Agent(object):
    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
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
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.

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
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def _on_train_begin(self):
        pass

    def _on_train_end(self):
        pass

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
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

    def _on_test_begin(self):
        pass

    def _on_test_end(self):
        pass

    def reset_states(self):
        pass

    def forward(self, observation):
        raise NotImplementedError()

    def backward(self, reward, terminal):
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        raise NotImplementedError()

    def load_weights(self, filepath):
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError()

    @property
    def metrics_names(self):
        return []


class Processor(object):
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processed observation will be stored in memory
        """
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_state_batch(self, batch):
        """Process for input into NN
        """
        return batch

    def process_action(self, action):
        return action

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []


class MultiInputProcessor(Processor):
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]


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
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
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
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
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
