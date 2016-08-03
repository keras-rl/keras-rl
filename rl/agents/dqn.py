from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.layers import Lambda, Input, merge, Layer
from keras.models import Model

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(Agent):
    def __init__(self, model, nb_actions, memory, window_length=1, policy=EpsGreedyQPolicy(),
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=10000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
                 custom_model_objects={}, processor=None):
        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, nb_actions))
        
        super(DQNAgent, self).__init__()

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_actions = nb_actions
        self.window_length = window_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.enable_double_dqn = enable_double_dqn
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.model = model
        self.memory = memory
        self.policy = policy
        self.policy._set_agent(self)
        self.processor = processor

        # State.
        self.compiled = False
        self.reset_states()

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        
        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
        
        def clipped_mse(y_true, y_pred):
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            return K.mean(K.square(delta), axis=-1)
        
        self.model.compile(optimizer=optimizer, loss=clipped_mse, metrics=metrics)
        
        self.compiled = True

    # TODO: implement support for pickle

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observations = deque(maxlen=self.window_length)

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_q_values(self, state):
        batch = self.process_state_batch([state])
        q_values = self.model.predict_on_batch(batch).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)

        # Select an action.
        while len(self.recent_observations) < self.recent_observations.maxlen:
            # Not enough data, fill the recent_observations queue with copies of the current input.
            # This allows us to immediately perform a policy action instead of falling back to random
            # actions.
            self.recent_observations.append(np.copy(observation))
        state = np.array(list(self.recent_observations)[1:] + [observation])
        assert len(state) == self.window_length
        q_values = self.compute_q_values(state)
        action = self.policy.select_action(q_values=q_values)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observations.append(observation)
        self.recent_action = action
        
        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if self.processor is not None:
            reward = self.processor.process_reward(reward)

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observations[-1], self.recent_action, reward, terminal)
        
        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size, self.window_length)
            assert len(experiences) == self.batch_size
            
            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal_batch = np.array(terminal_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            # Compute the current activations in the output layer given state0. This is hacky
            # since we do this in the training step anyway, but this is currently the simplest
            # way to set the gradients of the non-affected output units to zero.
            targets = self.model.predict_on_batch(state0_batch)
            assert targets.shape == (self.batch_size, self.nb_actions)

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for target, R, action in zip(targets, Rs, action_batch):
                target[action] = R  # update action with estimated accumulated reward
            targets = np.array(targets).astype('float32')

            # Finally, perform a single update on the entire batch.
            metrics = self.model.train_on_batch(state0_batch, targets)
            metrics += self.policy.run_metrics()

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        return self.model.metrics_names[:] + self.policy.metrics_names[:]


class NAFLayer(Layer):
    def __init__(self, nb_actions, **kwargs):
        self.nb_actions = nb_actions
        super(NAFLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # The input of this layer is [L, mu, a] in concatenated form. We first split
        # those up.
        idx = 0
        L_flat = x[:, idx:idx + (self.nb_actions * self.nb_actions + self.nb_actions) // 2]
        idx += (self.nb_actions * self.nb_actions + self.nb_actions) // 2
        mu = x[:, idx:idx + self.nb_actions]
        idx += self.nb_actions
        a = x[:, idx:idx + self.nb_actions]
        idx += self.nb_actions

        # Create L and L^T matrix, which we use to construct the positive-definite matrix P.
        L = None
        LT = None
        if K._BACKEND == 'theano':
            import theano.tensor as T
            import theano

            def fn(x, L_acc, LT_acc):
                x_ = K.zeros((self.nb_actions, self.nb_actions))
                x_ = T.set_subtensor(x_[np.tril_indices(self.nb_actions)], x)
                diag = K.exp(T.diag(x_))
                x_ = T.set_subtensor(x_[np.diag_indices(self.nb_actions)], diag)
                return x_, x_.T

            outputs_info = [
                K.zeros((self.nb_actions, self.nb_actions)),
                K.zeros((self.nb_actions, self.nb_actions)),
            ]
            results, _ = theano.scan(fn=fn, sequences=L_flat, outputs_info=outputs_info)
            L, LT = results
        elif K._BACKEND == 'tensorflow':
            import tensorflow as tf

            # Number of elements in a triangular matrix.
            nb_elems = (self.nb_actions * self.nb_actions + self.nb_actions) // 2

            # Create mask for the diagonal elements in L_flat. This is used to exponentiate
            # only the diagonal elements, which is done before gathering.
            diag_indeces = [0]
            for row in range(1, self.nb_actions):
                diag_indeces.append(diag_indeces[-1] + (row + 1))
            diag_mask = np.zeros(1 + nb_elems)  # +1 for the leading zero
            diag_mask[np.array(diag_indeces) + 1] = 1
            diag_mask = K.variable(diag_mask)

            # Add leading zero element to each element in the L_flat. We use this zero
            # element when gathering L_flat into a lower triangular matrix L.
            nb_rows = tf.shape(L_flat)[0]
            zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
            L_flat = tf.concat(1, [zeros, L_flat])
            
            # Create mask that can be used to gather elements from L_flat and put them
            # into a lower triangular matrix.
            tril_mask = np.zeros((self.nb_actions, self.nb_actions), dtype='int32')
            tril_mask[np.tril_indices(self.nb_actions)] = range(1, nb_elems + 1)
            
            # Finally, process each element of the batch.
            init = [
                K.zeros((self.nb_actions, self.nb_actions)),
                K.zeros((self.nb_actions, self.nb_actions)),
            ]
            
            def fn(a, x):
                # Exponentiate everything. This is much easier than only exponentiating
                # the diagonal elements, and, usually, the action space is relatively low.
                x_ = K.exp(x)
                # Only keep the diagonal elements.
                x_ *= diag_mask
                # Add the original, non-diagonal elements.
                x_ += x * (1. - diag_mask)
                # Finally, gather everything into a lower triangular matrix.
                L_ = tf.gather(x_, tril_mask)
                return [L_, tf.transpose(L_)]

            tmp = tf.scan(fn, L_flat, initializer=init)
            L = tmp[:, 0, :, :]
            LT = tmp[:, 1, :, :]
        else:
            raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
        assert L is not None
        assert LT is not None
        P = K.batch_dot(L, LT)
        assert K.ndim(P) == 3

        # Combine a, mu and P into a scalar (over the batches). What we compute here is
        # -.5 * (a - mu)^T * P * (a - mu), where * denotes the dot-product. Unfortunately
        # TensorFlow handles vector * P slightly suboptimal, hence we convert the vectors to
        # 1xd/dx1 matrices and finally flatten the resulting 1x1 matrix into a scalar. All
        # operations happen over the batch size, which is dimension 0.
        prod = K.batch_dot(K.expand_dims(a - mu, dim=1), P)
        prod = K.batch_dot(prod, K.expand_dims(a - mu, dim=-1))
        A = -.5 * K.batch_flatten(prod)
        assert K.ndim(A) == 2
        return A

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        if len(shape) != 2:
            raise RuntimeError('Input tensor must be 2D, has shape {} instead.'.format(input_shape))
        expected_elements = (self.nb_actions * self.nb_actions + self.nb_actions) // 2 + self.nb_actions + self.nb_actions
        if shape[-1] != expected_elements:
            raise RuntimeError(('Last dimension of input tensor must have exactly {} elements, ' +
                                'has {} elements instead. This layer expects the input in the ' + 
                                'following order: [L_flat, mu, action].').format(expected_elements, shape[-1]))
        shape[-1] = 1
        return tuple(shape)


class ContinuousDQNAgent(DQNAgent):
    def __init__(self, V_model, L_model, mu_model, nb_actions, memory, window_length=1,
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=10000, delta_range=(-np.inf, np.inf), custom_model_objects={},
                 processor=None, random_process=None):
        # TODO: Validate (important) input.
        
        # TODO: call super of abstract DQN agent

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_actions = nb_actions
        self.window_length = window_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.custom_model_objects = custom_model_objects
        self.random_process = random_process

        # Related objects.
        self.V_model = V_model
        self.L_model = L_model
        self.mu_model = mu_model
        self.memory = memory
        self.processor = processor

        # State.
        self.compiled = False
        self.reset_states()

    def update_target_model_hard(self):
        self.target_V_model.set_weights(self.V_model.get_weights())

    def load_weights(self, filepath):
        self.combined_model.load_weights(filepath)  # updates V, L and mu model since the weights are shared
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.combined_model.save_weights(filepath, overwrite=overwrite)

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # Create target V model. We don't need targets for mu or L.
        self.target_V_model = clone_model(self.V_model, self.custom_model_objects)
        self.target_V_model.compile(optimizer='sgd', loss='mse')

        # Build combined model.
        observation_shape = self.V_model.input._keras_shape[1:]
        a_in = Input(shape=(self.nb_actions,), name='action_input')
        o_in = Input(shape=observation_shape, name='observation_input')
        L_out = self.L_model([a_in, o_in])
        V_out = self.V_model(o_in)
        mu_out = self.mu_model(o_in)
        A_out = NAFLayer(self.nb_actions)(merge([L_out, mu_out, a_in], mode='concat'))
        combined_out = merge([A_out, V_out], mode='sum')
        combined = Model(input=[a_in, o_in], output=combined_out)

        # Compile combined model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_V_model, self.V_model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
        
        def clipped_mse(y_true, y_pred):
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            return K.mean(K.square(delta), axis=-1)
        
        combined.compile(loss=clipped_mse, optimizer=optimizer, metrics=metrics)
        self.combined_model = combined

        self.compiled = True

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.mu_model.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)

        # Select an action.
        while len(self.recent_observations) < self.recent_observations.maxlen:
            # Not enough data, fill the recent_observations queue with copies of the current input.
            # This allows us to immediately perform a policy action instead of falling back to random
            # actions.
            self.recent_observations.append(np.copy(observation))
        state = np.array(list(self.recent_observations)[1:] + [observation])
        assert len(state) == self.window_length
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observations.append(observation)
        self.recent_action = action
        
        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if self.processor is not None:
            reward = self.processor.process_reward(reward)

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observations[-1], self.recent_action, reward, terminal)
        
        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size, self.window_length)
            assert len(experiences) == self.batch_size
            
            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal_batch = np.array(terminal_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Compute Q values for mini-batch update.
            q_batch = self.target_V_model.predict_on_batch(state1_batch).flatten()
            assert q_batch.shape == (self.batch_size,)

            # Compute discounted reward.
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            assert Rs.shape == (self.batch_size,)

            # Finally, perform a single update on the entire batch.
            metrics = self.combined_model.train_on_batch([action_batch, state0_batch], Rs)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        return self.combined_model.metrics_names[:]
