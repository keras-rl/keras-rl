from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.layers import Lambda, Input, merge, Layer, TimeDistributed
from keras.models import Model

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class AbstractDQNAgent(Agent):
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=(-np.inf, np.inf), custom_model_objects={}, **kwargs):
        super(AbstractDQNAgent, self).__init__(**kwargs)

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
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_range': self.delta_range,
            'memory': get_object_config(self.memory),
        }

# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(AbstractDQNAgent):
    def __init__(self, model, policy=EpsGreedyQPolicy(), enable_double_dqn=True,
                 target_model=None, policy_model=None,
                 nb_max_steps_recurrent_unrolling=100, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape[-1] != self.nb_actions:
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Validate settings for recurrent DQN.
        self.is_recurrent = is_recurrent(model)
        if self.is_recurrent:
            if enable_double_dqn:
                raise ValueError('DoubleDQN (`enable_double_dqn = True`) is currently not supported for recurrent Q learning.')
            memory = kwargs['memory']
            if not memory.is_episodic:
                raise ValueError('Recurrent Q learning requires an episodic memory. You are trying to use it with memory={} instead.'.format(memory))
            if nb_max_steps_recurrent_unrolling and not model.stateful:
                raise ValueError('Recurrent Q learning with max. unrolling requires a stateful model.')
            if policy_model is None or not policy_model.stateful:
                raise ValueError('Recurrent Q learning requires a separate stateful policy model with batch_size=1. Please refer to an example to see how to properly set it up.')

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.nb_max_steps_recurrent_unrolling = nb_max_steps_recurrent_unrolling

        # Related objects.
        self.model = model
        self.target_model = target_model
        self.policy_model = policy_model if policy_model is not None else model
        self.policy = policy

        # State.
        self.reset_states()

    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['nb_max_steps_recurrent_unrolling'] = self.nb_max_steps_recurrent_unrolling
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        if self.target_model is None:
            self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')
        
        # Compile model.
        updates = []
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates += get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
        if self.policy_model is not self.model:
            # Update the policy model after every training step.
            updates += get_soft_target_model_updates(self.policy_model, self.model, 1.)
        if len(updates) > 0:
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_mse(args):
            y_true, y_pred, mask = args
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            delta *= mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)
            # Multiply by the number of actions to reverse the effect of the mean.
            loss *= float(self.nb_actions)
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        input_shape = (None, self.nb_actions) if self.is_recurrent else (self.nb_actions,)
        output_shape = (None, 1) if self.is_recurrent else (1,)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=input_shape)
        mask = Input(name='mask', shape=input_shape)
        loss_out = Lambda(clipped_masked_mse, output_shape=output_shape, name='loss')([y_pred, y_true, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.update_target_model_hard()
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()
            self.policy_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())
        if self.policy_model is not None:
            self.policy_model.set_weights(self.model.get_weights())

    def compute_q_values(self, state):
        batch = self.process_state_batch([state])
        if self.is_recurrent:
            # Add time axis.
            batch = batch.reshape((1,) + batch.shape)  # (1, 1, ...)
        q_values = self.policy_model.predict_on_batch(batch).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        action = self.policy.select_action(q_values=q_values)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action
        
        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            if self.is_recurrent:
                lengths = [len(seq) for seq in experiences]
                maxlen = np.max(lengths)

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = [[] for _ in range(len(experiences))]
                reward_batch = [[] for _ in range(len(experiences))]
                action_batch = [[] for _ in range(len(experiences))]
                terminal1_batch = [[] for _ in range(len(experiences))]
                state1_batch = [[] for _ in range(len(experiences))]
                for sequence_idx, sequence in enumerate(experiences):
                    for e in sequence:
                        state0_batch[sequence_idx].append(e.state0)
                        state1_batch[sequence_idx].append(e.state1)
                        reward_batch[sequence_idx].append(e.reward)
                        action_batch[sequence_idx].append(e.action)
                        terminal1_batch[sequence_idx].append(0. if e.terminal1 else 1.)

                    # Apply padding.
                    state_shape = state0_batch[sequence_idx][-1].shape
                    while len(state0_batch[sequence_idx]) < maxlen:
                        state0_batch[sequence_idx].append(np.zeros(state_shape))
                        state1_batch[sequence_idx].append(np.zeros(state_shape))
                        reward_batch[sequence_idx].append(0.)
                        action_batch[sequence_idx].append(0)
                        terminal1_batch[sequence_idx].append(1.)

                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                assert reward_batch.shape == (self.batch_size, maxlen)
                assert terminal1_batch.shape == reward_batch.shape
                assert len(action_batch) == len(reward_batch)
            else:
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
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                assert reward_batch.shape == (self.batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # Double DQN relies on the model for additional predictions, which we cannot use
                # since it must be stateful (we could save the state and re-apply, but this is
                # messy).
                assert not self.is_recurrent

                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                if self.is_recurrent:
                    self.model.reset_states()
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                if self.is_recurrent:
                    self.target_model.reset_states()
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                if self.is_recurrent:
                    assert target_q_values.shape == (self.batch_size, maxlen, self.nb_actions)
                else:
                    assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=-1)
            if self.is_recurrent:
                assert q_batch.shape == (self.batch_size, maxlen)
            else:
                q_batch = q_batch.flatten()
                assert q_batch.shape == (self.batch_size,)

            if self.is_recurrent:
                targets = np.zeros((self.batch_size, maxlen, self.nb_actions))
                dummy_targets = np.zeros((self.batch_size, maxlen, 1))
                masks = np.zeros((self.batch_size, maxlen, self.nb_actions))
            else:
                targets = np.zeros((self.batch_size, self.nb_actions))
                dummy_targets = np.zeros((self.batch_size,))
                masks = np.zeros((self.batch_size, self.nb_actions))
            
            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            if self.is_recurrent:
                for batch_idx, (inner_targets, inner_masks, inner_Rs, inner_action_batch, length) in enumerate(zip(targets, masks, Rs, action_batch, lengths)):
                    for idx, (target, mask, R, action) in enumerate(zip(inner_targets, inner_masks, inner_Rs, inner_action_batch)):
                        target[action] = R  # update action with estimated accumulated reward
                        dummy_targets[batch_idx, idx] = R
                        if idx < length:  # only enable loss for valid transitions
                            mask[action] = 1.  # enable loss for this specific action

            else:
                for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                    target[action] = R  # update action with estimated accumulated reward
                    dummy_targets[idx] = R
                    mask[action] = 1.  # enable loss for this specific action

            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch

            # In the recurrent case, we support splitting the sequences into multiple
            # chunks. Each chunk is then used as a training example. The reason for this is that,
            # for too long episodes, the unrolling in time during backpropagation can exceed the
            # memory of the GPU (or, to a lesser degree, the RAM if training on CPU).
            if self.is_recurrent and self.nb_max_steps_recurrent_unrolling:
                assert targets.ndim == 3
                steps = targets.shape[1]  # (batch_size, steps, actions)
                nb_chunks = int(np.ceil(float(steps) / float(self.nb_max_steps_recurrent_unrolling)))
                chunks = []
                for chunk_idx in range(nb_chunks):
                    start = chunk_idx * self.nb_max_steps_recurrent_unrolling
                    t = targets[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    m = masks[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    iss = [i[:, start:start + self.nb_max_steps_recurrent_unrolling, ...] for i in ins]
                    dt = dummy_targets[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    chunks.append((iss, t, m, dt))
            else:
                chunks = [(ins, targets, masks, dummy_targets)]

            metrics = []
            if self.is_recurrent:
                # Reset states before training on the entire sequence.
                self.trainable_model.reset_states()
            for i, t, m, dt in chunks:
                # Finally, perform a single update on the entire batch. We use a dummy target since
                # the actual loss is computed in a Lambda layer that needs more complex input. However,
                # it is still useful to know the actual target to compute metrics properly.
                ms = self.trainable_model.train_on_batch(i + [t, m], [dt, t])
                ms = [metric for idx, metric in enumerate(ms) if idx not in (1, 2)]  # throw away individual losses
                metrics.append(ms)
            metrics = np.mean(metrics, axis=0).tolist()
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)


class NAFLayer(Layer):
    def __init__(self, nb_actions, mode='full', **kwargs):
        if mode not in ('full', 'diag'):
            raise RuntimeError('Unknown mode "{}" in NAFLayer.'.format(self.mode))

        self.nb_actions = nb_actions
        self.mode = mode
        super(NAFLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # TODO: validate input shape

        # The input of this layer is [L, mu, a] in concatenated form. We first split
        # those up.
        idx = 0
        if self.mode == 'full':
            L_flat = x[:, idx:idx + (self.nb_actions * self.nb_actions + self.nb_actions) // 2]
            idx += (self.nb_actions * self.nb_actions + self.nb_actions) // 2
        elif self.mode == 'diag':
            L_flat = x[:, idx:idx + self.nb_actions]
            idx += self.nb_actions
        else:
            L_flat = None
        assert L_flat is not None
        mu = x[:, idx:idx + self.nb_actions]
        idx += self.nb_actions
        a = x[:, idx:idx + self.nb_actions]
        idx += self.nb_actions

        if self.mode == 'full':
            # Create L and L^T matrix, which we use to construct the positive-definite matrix P.
            L = None
            LT = None
            if K._BACKEND == 'theano':
                import theano.tensor as T
                import theano

                def fn(x, L_acc, LT_acc):
                    x_ = K.zeros((self.nb_actions, self.nb_actions))
                    x_ = T.set_subtensor(x_[np.tril_indices(self.nb_actions)], x)
                    diag = K.exp(T.diag(x_) + K.epsilon())
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
                    x_ = K.exp(x + K.epsilon())
                    # Only keep the diagonal elements.
                    x_ *= diag_mask
                    # Add the original, non-diagonal elements.
                    x_ += x * (1. - diag_mask)
                    # Finally, gather everything into a lower triangular matrix.
                    L_ = tf.gather(x_, tril_mask)
                    return [L_, tf.transpose(L_)]

                tmp = tf.scan(fn, L_flat, initializer=init)
                if isinstance(tmp, (list, tuple)):
                    # TensorFlow 0.10 now returns a tuple of tensors.
                    L, LT = tmp
                else:
                    # Old TensorFlow < 0.10 returns a shared tensor.
                    L = tmp[:, 0, :, :]
                    LT = tmp[:, 1, :, :]
            else:
                raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
            assert L is not None
            assert LT is not None
            P = K.batch_dot(L, LT)
        elif self.mode == 'diag':
            if K._BACKEND == 'theano':
                import theano.tensor as T
                import theano

                def fn(x, P_acc):
                    x_ = K.zeros((self.nb_actions, self.nb_actions))
                    x_ = T.set_subtensor(x_[np.diag_indices(self.nb_actions)], x)
                    return x_

                outputs_info = [
                    K.zeros((self.nb_actions, self.nb_actions)),
                ]
                P, _ = theano.scan(fn=fn, sequences=L_flat, outputs_info=outputs_info)
            elif K._BACKEND == 'tensorflow':
                import tensorflow as tf

                # Create mask that can be used to gather elements from L_flat and put them
                # into a diagonal matrix.
                diag_mask = np.zeros((self.nb_actions, self.nb_actions), dtype='int32')
                diag_mask[np.diag_indices(self.nb_actions)] = range(1, self.nb_actions + 1)

                # Add leading zero element to each element in the L_flat. We use this zero
                # element when gathering L_flat into a lower triangular matrix L.
                nb_rows = tf.shape(L_flat)[0]
                zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
                L_flat = tf.concat(1, [zeros, L_flat])

                # Finally, process each element of the batch.
                def fn(a, x):
                    x_ = tf.gather(x, diag_mask)
                    return x_

                P = tf.scan(fn, L_flat, initializer=K.zeros((self.nb_actions, self.nb_actions)))
            else:
                raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
        assert P is not None
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
        if self.mode == 'full':
            expected_elements = (self.nb_actions * self.nb_actions + self.nb_actions) // 2 + self.nb_actions + self.nb_actions
        elif self.mode == 'diag':
            expected_elements = self.nb_actions + self.nb_actions + self.nb_actions
        else:
            expected_elements = None
        assert expected_elements is not None
        if shape[-1] != expected_elements:
            raise RuntimeError(('Last dimension of input tensor must have exactly {} elements, ' +
                                'has {} elements instead. This layer expects the input in the ' + 
                                'following order: [L_flat, mu, action].').format(expected_elements, shape[-1]))
        shape[-1] = 1
        return tuple(shape)


class ContinuousDQNAgent(AbstractDQNAgent):
    def __init__(self, V_model, L_model, mu_model, random_process=None,
                 covariance_mode='full', *args, **kwargs):
        super(ContinuousDQNAgent, self).__init__(*args, **kwargs)

        # TODO: Validate (important) input.

        # Parameters.
        self.random_process = random_process
        self.covariance_mode = covariance_mode

        # Related objects.
        self.V_model = V_model
        self.L_model = L_model
        self.mu_model = mu_model

        # State.
        self.reset_states()

    def update_target_model_hard(self):
        self.target_V_model.set_weights(self.V_model.get_weights())

    def load_weights(self, filepath):
        self.combined_model.load_weights(filepath)  # updates V, L and mu model since the weights are shared
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.combined_model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.combined_model.reset_states()
            self.target_V_model.reset_states()

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # Create target V model. We don't need targets for mu or L.
        self.target_V_model = clone_model(self.V_model, self.custom_model_objects)
        self.target_V_model.compile(optimizer='sgd', loss='mse')

        # Build combined model.
        a_in = Input(shape=(self.nb_actions,), name='action_input')
        if type(self.V_model.input) is list:
            observation_shapes = [i._keras_shape[1:] for i in self.V_model.input]
        else:
            observation_shapes = [self.V_model.input._keras_shape[1:]]
        os_in = [Input(shape=shape, name='observation_input_{}'.format(idx)) for idx, shape in enumerate(observation_shapes)]
        L_out = self.L_model([a_in] + os_in)
        V_out = self.V_model(os_in)
        mu_out = self.mu_model(os_in)
        A_out = NAFLayer(self.nb_actions, mode=self.covariance_mode)(merge([L_out, mu_out, a_in], mode='concat'))
        combined_out = merge([A_out, V_out], mode='sum')
        combined = Model(input=[a_in] + os_in, output=combined_out)

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
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action
        
        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics
        
        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size
            
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
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Compute Q values for mini-batch update.
            q_batch = self.target_V_model.predict_on_batch(state1_batch).flatten()
            assert q_batch.shape == (self.batch_size,)

            # Compute discounted reward.
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            assert Rs.shape == (self.batch_size,)

            # Finally, perform a single update on the entire batch.
            if len(self.combined_model.input) == 2:
                metrics = self.combined_model.train_on_batch([action_batch, state0_batch], Rs)
            else:
                metrics = self.combined_model.train_on_batch([action_batch] + state0_batch, Rs)
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def get_config(self):
        config = super(ContinuousDQNAgent, self).get_config()
        config['V_model'] = get_object_config(self.V_model)
        config['mu_model'] = get_object_config(self.mu_model)
        config['L_model'] = get_object_config(self.L_model)
        if self.compiled:
            config['target_V_model'] = get_object_config(self.target_V_model)
        return config

    @property
    def metrics_names(self):
        names = self.combined_model.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names
