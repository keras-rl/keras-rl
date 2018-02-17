from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList, TensorBoard, CSVLogger
from keras.utils.generic_utils import Progbar
# from rl.core import Agent


class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        pass

    def on_step_begin(self, step, logs={}):
        pass

    def on_step_end(self, step, logs={}):
        pass

    def on_action_begin(self, action, logs={}):
        pass

    def on_action_end(self, action, logs={}):
        pass


class CallbackList(KerasCallbackList):
    def set_model(self, model):
        from rl.core import Agent
        for callback in self.callbacks:
            if issubclass(type(model), Agent) and \
                    (issubclass(type(callback), TensorBoard)):
                        callback.set_model(model.model)
            else:
                callback.set_model(model)

    def _set_env(self, env):
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_begin` callback.
            # If not, fall back to `on_epoch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_end` callback.
            # If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_begin` callback.
            # If not, fall back to `on_batch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_end` callback.
            # If not, fall back to `on_batch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)


class TestLogger(Callback):
    def on_train_begin(self, logs):
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs):
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        print(template.format(*variables))


class TrainEpisodeLogger(Callback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))
        
    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]          
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


class TrainIntervalLogger(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reset()

    def reset(self):
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.metrics = []
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:.3f}'.format(name, mean)
                
                formatted_infos = ''
                if len(self.infos) > 0:
                    infos = np.array(self.infos)
                    if not np.isnan(infos).all():  # not all values are means
                        means = np.nanmean(self.infos, axis=0)
                        assert means.shape == (len(self.info_names),)
                        for name, mean in zip(self.info_names, means):
                            formatted_infos += ' - {}: {:.3f}'.format(name, mean)
                print('{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]{}{}'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards), formatted_metrics, formatted_infos))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))

    def on_step_end(self, step, logs):
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]
        self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])

    def on_episode_end(self, episode, logs):
        self.episode_rewards.append(logs['episode_reward'])


class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        self.save_data()

    def on_episode_begin(self, episode, logs):
        assert episode not in self.metrics
        assert episode not in self.starts
        self.metrics[episode] = []
        self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.starts[episode]
        
        metrics = self.metrics[episode]
        if np.isnan(metrics).all():
            mean_metrics = np.array([np.nan for _ in self.metrics_names])
        else:
            mean_metrics = np.nanmean(metrics, axis=0)
        assert len(mean_metrics) == len(self.metrics_names)

        data = list(zip(self.metrics_names, mean_metrics))
        data += list(logs.items())
        data += [('episode', episode), ('duration', duration)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()

        # Clean up.
        del self.metrics[episode]
        del self.starts[episode]

    def on_step_end(self, step, logs):
        self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)


class Visualizer(Callback):
    def on_action_end(self, action, logs):
        self.env.render(mode='human')


class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True)


class RlTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 agent=None):
        super(RlTensorBoard, self).__init__(log_dir, histogram_freq, batch_size, write_graph, write_grads,
                                            write_images, embeddings_freq, embeddings_layer_names,
                                            embeddings_metadata)
        self.agent = agent

    def set_validation_data(self):
        import numpy as np
        experiences = self.agent.memory.sample(self.agent.batch_size)
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.agent.process_state_batch(state0_batch)
        state1_batch = self.agent.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.agent.batch_size, self.agent.nb_actions)

        # Compute Q values for mini-batch update.
        q_batch = self.agent.target_V_model.predict_on_batch(state1_batch).flatten()
        assert q_batch.shape == (self.batch_size,)

        # Compute discounted reward.
        discounted_reward_batch = self.agent.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        assert Rs.shape == (self.agent.batch_size,)

        self.validation_data = [action_batch, state0_batch, np.expand_dims(Rs, 1), np.ones(self.agent.batch_size)]

    def on_epoch_end(self, epoch, logs=None):
        self.set_validation_data()
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            try:
                summary_value.simple_value = value.item()
            except:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()
