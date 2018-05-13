from __future__ import division
import warnings

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class AbstractDQNAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
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

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
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

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }

# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(AbstractDQNAgent):
    """
    # Arguments 
        model__: A Keras model. 
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py). 
        test_policy__: A Keras-rl policy. 
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting. 
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al. 
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581). 
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta))) 
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta))) 
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta) 
 
    """
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, self.nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()

    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

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

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

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
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
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
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

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

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)


class NAFLayer(Layer):
    """Write me
    """
    def __init__(self, nb_actions, mode='full', **kwargs):
        if mode not in ('full', 'diag'):
            raise RuntimeError('Unknown mode "{}" in NAFLayer.'.format(self.mode))

        self.nb_actions = nb_actions
        self.mode = mode
        super(NAFLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # TODO: validate input shape

        assert (len(x) == 3)
        L_flat = x[0]
        mu = x[1]
        a = x[2]

        if self.mode == 'full':
            # Create L and L^T matrix, which we use to construct the positive-definite matrix P.
            L = None
            LT = None
            if K.backend() == 'theano':
                import theano.tensor as T
                import theano

                def fn(x, L_acc, LT_acc):
                    x_ = K.zeros((self.nb_actions, self.nb_actions))
                    x_ = T.set_subtensor(x_[np.tril_indices(self.nb_actions)], x)
                    diag = K.exp(T.diag(x_)) + K.epsilon()
                    x_ = T.set_subtensor(x_[np.diag_indices(self.nb_actions)], diag)
                    return x_, x_.T

                outputs_info = [
                    K.zeros((self.nb_actions, self.nb_actions)),
                    K.zeros((self.nb_actions, self.nb_actions)),
                ]
                results, _ = theano.scan(fn=fn, sequences=L_flat, outputs_info=outputs_info)
                L, LT = results
            elif K.backend() == 'tensorflow':
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
                try:
                    # Old TF behavior.
                    L_flat = tf.concat(1, [zeros, L_flat])
                except TypeError:
                    # New TF behavior
                    L_flat = tf.concat([zeros, L_flat], 1)

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
                    x_ = K.exp(x) + K.epsilon()
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
                raise RuntimeError('Unknown Keras backend "{}".'.format(K.backend()))
            assert L is not None
            assert LT is not None
            P = K.batch_dot(L, LT)
        elif self.mode == 'diag':
            if K.backend() == 'theano':
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
            elif K.backend() == 'tensorflow':
                import tensorflow as tf

                # Create mask that can be used to gather elements from L_flat and put them
                # into a diagonal matrix.
                diag_mask = np.zeros((self.nb_actions, self.nb_actions), dtype='int32')
                diag_mask[np.diag_indices(self.nb_actions)] = range(1, self.nb_actions + 1)

                # Add leading zero element to each element in the L_flat. We use this zero
                # element when gathering L_flat into a lower triangular matrix L.
                nb_rows = tf.shape(L_flat)[0]
                zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
                try:
                    # Old TF behavior.
                    L_flat = tf.concat(1, [zeros, L_flat])
                except TypeError:
                    # New TF behavior
                    L_flat = tf.concat([zeros, L_flat], 1)

                # Finally, process each element of the batch.
                def fn(a, x):
                    x_ = tf.gather(x, diag_mask)
                    return x_

                P = tf.scan(fn, L_flat, initializer=K.zeros((self.nb_actions, self.nb_actions)))
            else:
                raise RuntimeError('Unknown Keras backend "{}".'.format(K.backend()))
        assert P is not None
        assert K.ndim(P) == 3

        # Combine a, mu and P into a scalar (over the batches). What we compute here is
        # -.5 * (a - mu)^T * P * (a - mu), where * denotes the dot-product. Unfortunately
        # TensorFlow handles vector * P slightly suboptimal, hence we convert the vectors to
        # 1xd/dx1 matrices and finally flatten the resulting 1x1 matrix into a scalar. All
        # operations happen over the batch size, which is dimension 0.
        prod = K.batch_dot(K.expand_dims(a - mu, 1), P)
        prod = K.batch_dot(prod, K.expand_dims(a - mu, -1))
        A = -.5 * K.batch_flatten(prod)
        assert K.ndim(A) == 2
        return A

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise RuntimeError("Expects 3 inputs: L, mu, a")
        for i, shape in enumerate(input_shape):
            if len(shape) != 2:
                raise RuntimeError("Input {} has {} dimensions but should have 2".format(i, len(shape)))
        assert self.mode in ('full','diag')
        if self.mode == 'full':
            expected_elements = (self.nb_actions * self.nb_actions + self.nb_actions) // 2
        elif self.mode == 'diag':
            expected_elements = self.nb_actions
        else:
            expected_elements = None
        assert expected_elements is not None
        if input_shape[0][1] != expected_elements:
            raise RuntimeError("Input 0 (L) should have {} elements but has {}".format(input_shape[0][1]))
        if input_shape[1][1] != self.nb_actions:
            raise RuntimeError(
                "Input 1 (mu) should have {} elements but has {}".format(self.nb_actions, input_shape[1][1]))
        if input_shape[2][1] != self.nb_actions:
            raise RuntimeError(
                "Input 2 (action) should have {} elements but has {}".format(self.nb_actions, input_shape[1][1]))
        return input_shape[0][0], 1


class NAFAgent(AbstractDQNAgent):
    """Write me
    """
    def __init__(self, V_model, L_model, mu_model, random_process=None,
                 covariance_mode='full', *args, **kwargs):
        super(NAFAgent, self).__init__(*args, **kwargs)

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
        A_out = NAFLayer(self.nb_actions, mode=self.covariance_mode)([L_out, mu_out, a_in])
        combined_out = Lambda(lambda x: x[0]+x[1], output_shape=lambda x: x[0])([A_out, V_out])
        combined = Model(inputs=[a_in] + os_in, outputs=[combined_out])
        # Compile combined model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_V_model, self.V_model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        combined.compile(loss=clipped_error, optimizer=optimizer, metrics=metrics)
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

    @property
    def layers(self):
        return self.combined_model.layers[:]

    def get_config(self):
        config = super(NAFAgent, self).get_config()
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


# Aliases
ContinuousDQNAgent = NAFAgent
