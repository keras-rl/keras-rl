from __future__ import division
import warnings
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.utils.generic_utils import Progbar
from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.memory import PrioritizedMemory, PartitionedMemory
from rl.layers import *

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

    def _calc_double_q_values(self, state1_batch):
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

        return q_batch

    def _build_dueling_arch(self, model, dueling_type):
        #bulid the two-stream architecture
        layer = model.layers[-2]
        nb_action = model.output._keras_shape[-1]
        y = Dense(nb_action + 1, activation='linear')(layer.output)
        #preserve use of noisy nets.
        if isinstance(layer, NoisyNetDense):
            y = NoisyNetDense(nb_action + 1, activation='linear')(layer.output)
        # options for dual-stream merger
        if dueling_type == 'avg':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
        elif dueling_type == 'max':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
        elif dueling_type == 'naive':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        return Model(inputs=model.input, outputs=outputlayer)

# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgent(AbstractDQNAgent):
    """
    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enables target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enables dueling architecture proposed by Wang et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
        nb_actions__: The total number of actions the agent can take. Dependent on the environment.
        processor__: A Keras-rl processor. An intermediary between the environment and the agent. Resizes the input, clips rewards etc. Similar to gym env wrappers.
        nb_steps_warmup__: An integer number of random steps to take before learning begins. This puts experience into the memory.
        gamma__: The discount factor of future rewards in the Q function.
        target_model_update__: How often to update the target model. Longer intervals stabilize training.
        train_interval__: The integer number of steps between each learning process.
        delta_clip__: A component of the huber loss.
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
            model = self._build_dueling_arch(model, dueling_type)
        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.reset_states()

        #flag for changes to algorithm that come from dealing with importance sampling weights and priorities
        self.prioritized = True if isinstance(self.memory, PrioritizedMemory) else False
        
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
            y_true, y_pred, importance_weights, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            # adjust updates by importance weights. Note that importance weights are just 1.0
            # (and have no effect) if not using a prioritized memory
            return K.sum(loss * importance_weights, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        importance_weights = Input(name='importance_weights',shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, importance_weights, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, importance_weights, mask], outputs=[loss_out, y_pred])
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

            if self.prioritized:
                # Calculations for current beta value based on a linear schedule.
                current_beta = self.memory.calculate_beta(self.step)
                # Sample from the memory.
                experiences = self.memory.sample(self.batch_size, current_beta)
            else:
                #SequentialMemory
                experiences = self.memory.sample(self.batch_size)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            # We will be updating the idxs of the priority trees with new priorities
            pr_idxs = []

            if self.prioritized:
                for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)
                importance_weights = experiences[-2]
                pr_idxs = experiences[-1]
            else: #SequentialMemory
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
                q_batch = self._calc_double_q_values(state1_batch)
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
            #Putting together the multi-step target
            Rs = reward_batch + discounted_reward_batch

            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            if not self.prioritized:
                importance_weights = [1. for _ in range(self.batch_size)]
            #Make importance_weights the same shape as the other tensors that are passed into the trainable model
            assert len(importance_weights) == self.batch_size
            importance_weights = np.array(importance_weights)
            importance_weights = np.vstack([importance_weights]*self.nb_actions)
            importance_weights = np.reshape(importance_weights, (self.batch_size, self.nb_actions))
            # Perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, importance_weights, masks], [dummy_targets, targets])

            if self.prioritized:
                assert len(pr_idxs) == self.batch_size
                #Calculate new priorities.
                y_true = targets
                y_pred = self.model.predict_on_batch(ins)
                #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
                new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + 1e-5
                assert len(new_priorities) == self.batch_size
                #update priorities
                self.memory.update_priorities(pr_idxs, new_priorities)

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

class DQfDAgent(AbstractDQNAgent):

    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=True,
                 dueling_type='avg', n_step=10, pretraining_steps=750000, large_margin=.8, lam_2=1., *args, **kwargs):
        """
        Deep Q-Learning from Demonstrations. Uses demonstrations from an expert controller to kickstart training and improve
        sample efficiency. [paper](https://arxiv.org/abs/1704.03732).

        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enables target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enables dueling architecture proposed by Wang et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
        nb_actions__: The total number of actions the agent can take. Dependent on the environment.
        processor__: A Keras-rl processor. An intermediary between the environment and the agent. Resizes the input, clips rewards etc. Similar to gym env wrappers.
        nb_steps_warmup__: An integer number of random steps to take before learning begins. This puts experience into the memory.
        gamma__: The discount factor of future rewards in the Q function.
        target_model_update__: How often to update the target model. Longer intervals stabilize training.
        train_interval__: The integer number of steps between each learning process.
        delta_clip__: A component of the huber loss.
        n_step__: exponent for multi-step learning. Larger values extend the future reward approximations further into the future.
        pretraining_steps__: Length of 'pretraining' in which the agent learns exclusively from the expert demonstration data.
        large_margin__: Constant value that pushes loss of incorrect action choices a margin higher than the others.
        lam_2__: Imitation loss coefficient.
        """

        super(DQfDAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQfD expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, self.nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQfD expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            model = self._build_dueling_arch(model, dueling_type)
        self.model = model

        #multi-step learning parameter.
        self.n_step = n_step
        self.pretraining_steps = pretraining_steps
        self.pretraining = True
        #margin to add when action of agent != action of expert
        self.large_margin = large_margin
        #coefficient of supervised loss component of the loss function
        self.lam_2 = lam_2
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.reset_states()

        assert isinstance(self.memory, PartitionedMemory), "DQfD needs a PartitionedMemory to store expert transitions without overwriting them."
        assert len(self.memory.observations) > 0, "Pre-load the memory with demonstration data."

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model; optimizer and loss choices are arbitrary
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def dqfd_error(args):
            y_true, y_true_n, y_pred, importance_weights, agent_actions, large_margin, lam_2, mask = args
            #Standard DQN loss
            j_dq = huber_loss(y_true, y_pred, self.delta_clip) * mask
            j_dq *= importance_weights
            j_dq = K.sum(j_dq, axis=-1)
            #N-step DQN loss
            j_n = huber_loss(y_true_n, y_pred, self.delta_clip) * mask
            j_n *= importance_weights
            j_n = K.sum(j_n, axis=-1)
            #Large margin supervised classification loss
            Q_a = y_pred * agent_actions
            Q_ae = y_pred * mask
            j_e =  lam_2 * (Q_a + large_margin - Q_ae)
            j_e = K.sum(j_e, axis=-1)
            # in Keras, j_l2 from the paper is implemented as a part of the network itself (using regularizers.l2)
            return j_dq + j_n + j_e

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        y_true_n = Input(name='y_true_n', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        importance_weights = Input(name='importance_weights',shape=(self.nb_actions,))
        agent_actions = Input(name='agent_actions',shape=(self.nb_actions,))
        large_margin = Input(name='large-margin',shape=(self.nb_actions,))
        lam_2 = Input(name='lam_2',shape=(self.nb_actions,))
        loss_out = Lambda(dqfd_error, output_shape=(1,), name='loss')([y_true, y_true_n, y_pred, importance_weights, agent_actions, large_margin, lam_2, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, y_true_n, importance_weights, agent_actions, large_margin, lam_2, mask], outputs=[loss_out, y_pred])
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

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        progbar = Progbar(self.pretraining_steps, interval=.1)
        print("Pretraining for {} steps...".format(self.pretraining_steps))
        for step in range(self.pretraining_steps):
            self.backward(0, False)
            progbar.update(step)
        self.pretraining = False

        super(DQfDAgent, self).fit(env, nb_steps, action_repetition, callbacks, verbose,
                visualize, nb_max_start_steps, start_step_policy, log_interval,
                nb_max_episode_steps)

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0 and not self.pretraining:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training and not self.pretraining:
            return metrics

        # Train the network on a single stochastic batch.
        if (self.step % self.train_interval == 0) or self.pretraining:
            # Calculations for current beta value based on a linear schedule.
            current_beta = self.memory.calculate_beta(self.step)
            # Sample from the memory.
            idxs = self.memory.sample_proportional(self.batch_size)
            experiences_n = self.memory.sample_by_idxs(idxs, self.batch_size, current_beta, self.n_step, self.gamma)
            experiences = self.memory.sample_by_idxs(idxs, self.batch_size, current_beta)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
            importance_weights = experiences[-2]

            # Get n-step versions. The state batch is the observation n steps
            # after the state0 batch (or the first terminal state, whichever is first).
            # The reward batch is the sum of discounted rewards between state0 and
            # the state in the state_batch_n. Terminal batch is used to eliminate
            # the target network's q values when the discounted rewards already extend
            # to the end of the episode.
            state_batch_n = []
            reward_batch_n = []
            terminal_batch_n = []
            for e in experiences_n[:-2]:
                state_batch_n.append(e.state1)
                reward_batch_n.append(e.reward)
                terminal_batch_n.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            state_batch_n = self.process_state_batch(state_batch_n)
            terminal1_batch = np.array(terminal1_batch)
            terminal_batch_n = np.array(terminal_batch_n)
            reward_batch = np.array(reward_batch)
            reward_batch_n = np.array(reward_batch_n)
            assert reward_batch.shape == (self.batch_size,)
            assert reward_batch_n.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                q_batch = self._calc_double_q_values(state1_batch)
                q_batch_n = self._calc_double_q_values(state_batch_n)
            else:
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
                # Repeat this process for the n-step state.
                target_q_values_n = self.target_model.predict_on_batch(state_batch_n)
                assert target_q_values_n.shape == (self.batch_size, self.nb_actions)
                q_batch_n = np.max(target_q_values_n, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)
            assert q_batch_n.shape == (self.batch_size,)

            #Multi-step loss targets
            targets_n = np.zeros((self.batch_size, self.nb_actions))
            masks = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets_n = np.zeros((self.batch_size,))
            discounted_reward_batch_n = (self.gamma**self.n_step) * q_batch_n
            discounted_reward_batch_n *= terminal_batch_n
            assert discounted_reward_batch_n.shape == reward_batch_n.shape
            Rs_n = reward_batch_n + discounted_reward_batch_n
            for idx, (target, mask, R, action) in enumerate(zip(targets_n, masks, Rs_n, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets_n[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets_n = np.array(targets_n).astype('float32')

            #Single-step loss targets
            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            discounted_reward_batch = (self.gamma) * q_batch
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            #Make importance_weights the same shape as the other tensors that are passed into the trainable model
            assert len(importance_weights) == self.batch_size
            importance_weights = np.array(importance_weights)
            importance_weights = np.vstack([importance_weights]*self.nb_actions)
            importance_weights = np.reshape(importance_weights, (self.batch_size, self.nb_actions))

            #we need the network to make its own decisions for each of the expert's transitions (so we can compare)
            y_pred = self.model.predict_on_batch(state0_batch)
            agent_actions = np.argmax(y_pred, axis=1)
            assert agent_actions.shape == (self.batch_size,)
            #one-hot encode actions, gives the shape needed to pass into the model
            agent_actions = np.eye(self.nb_actions)[agent_actions]
            expert_actions = masks
            #l is the large margin term, which skews loss function towards incorrect imitations
            large_margin = np.zeros_like(expert_actions, dtype='float32')
            #lambda_2 is used to eliminate supervised loss for self-generated transitions
            lam_2 = np.zeros_like(expert_actions, dtype='float32')

            # Here we are building the large margin term, which is a matrix
            # with a postiive entry where the agent and expert actions differ
            for i, idx in enumerate(idxs):
                if idx < self.memory.permanent_idx:
                    #this is an expert demonstration, enable supervised loss
                    lam_2[i,:] = self.lam_2
                    for j in range(agent_actions.shape[1]):
                        if agent_actions[i,j] == 1:
                            if expert_actions[i,j] != 1:
                                #if agent and expert had different predictions, increase l
                                large_margin[i,j] = self.large_margin

            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, targets_n, importance_weights, agent_actions, large_margin, lam_2, masks], [dummy_targets, targets])

            assert len(idxs) == self.batch_size
            #Calculate new priorities.
            y_true = targets
            #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
            #Boost for expert transitions is handled in memory.PartitionedMemory.update_priorities
            new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + .001
            assert len(new_priorities) == self.batch_size
            self.memory.update_priorities(idxs, new_priorities)

            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def get_config(self):
        config = super(DQfDAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        config['pretraining_steps'] = self.pretraining_steps
        config['n_step'] = self.n_step
        config['large_margin'] = self.large_margin
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

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


# Aliases
ContinuousDQNAgent = NAFAgent
