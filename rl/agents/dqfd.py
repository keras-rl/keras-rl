from __future__ import division
import warnings
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense

from rl.core import Agent
from rl.agents.dqn import AbstractDQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.memory import PartitionedMemory
from rl.layers import *

class DQfDAgent(AbstractDQNAgent):

    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=True,
                 dueling_type='avg', n_step=10, pretraining_steps=750000, large_margin=.8, lam_2=1., *args, **kwargs):

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
            #bulid the two-stream architecture
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # options for dual-stream merger
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"
            model = Model(inputs=model.input, outputs=outputlayer)
        self.model = model

        #multi-step learning parameter.
        self.n_step = n_step
        self.pretraining_steps = pretraining_steps
        #margin to add when action of expert != action of agent
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

        assert type(self.memory) == PartitionedMemory, "DQfD needs a PartitionedMemory to store expert transitions without overwriting them."
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
            y_true, y_true_n, y_pred, importance_weights, agent_actions, l, lam_2, mask = args
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
            j_e = K.square(lam_2 * (Q_a + l - Q_ae))
            j_e = K.sum(j_e, axis=-1)
            # in Keras, j_l2 from the paper is implemented as a part of the network itself (using regularizers.l2 with a value of 1e-4)
            return j_dq + j_n + j_e

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        y_true_n = Input(name='y_true_n', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        importance_weights = Input(name='importance_weights',shape=(self.nb_actions,))
        agent_actions = Input(name='agent_actions',shape=(self.nb_actions,))
        l = Input(name='large-margin-classification',shape=(self.nb_actions,))
        lam_2 = Input(name='lam_2',shape=(self.nb_actions,))
        loss_out = Lambda(dqfd_error, output_shape=(1,), name='loss')([y_true, y_true_n, y_pred, importance_weights, agent_actions, l, lam_2, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, y_true_n, importance_weights, agent_actions, l, lam_2, mask], outputs=[loss_out, y_pred])
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
        if self.step % self.memory_interval == 0 and self.step > self.pretraining_steps:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            return metrics

        # Train the network on a single stochastic batch.
        if self.step % self.train_interval == 0:

            # Calculations for current beta value based on a linear schedule.
            current_beta = self.memory.calculate_beta(self.step)
            # Sample from the memory.
            experiences = self.memory.sample(self.batch_size, current_beta)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            # We will be updating the idxs of the priority tree with new priorities
            pr_idxs = []
            for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
            importance_weights = experiences[-2]
            pr_idxs = experiences[-1]

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
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            #Multi-step loss targets
            targets_n = np.zeros((self.batch_size, self.nb_actions))
            masks = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets_n = np.zeros((self.batch_size,))
            discounted_reward_batch_n = (self.gamma **(self.n_step)) * q_batch
            discounted_reward_batch_n *= terminal1_batch
            assert discounted_reward_batch_n.shape == reward_batch.shape
            Rs_n = (reward_batch **(self.n_step)) + discounted_reward_batch_n
            for idx, (target, mask, R, action) in enumerate(zip(targets_n, masks, Rs_n, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets_n[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets_n = np.array(targets_n).astype('float32')
            masks = np.array(masks).astype('float32')

            #Single-step loss targets
            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            discounted_reward_batch = (self.gamma) * q_batch
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = (reward_batch) + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
            targets = np.array(targets).astype('float32')

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
            l = np.zeros_like(expert_actions, dtype='float32')
            #lambda_2 is used to eliminate supervised loss for self-generated transitions
            lam_2 = np.zeros_like(expert_actions, dtype='float32')

            for i, idx in enumerate(pr_idxs):
                if idx < self.memory.permanent_idx:
                    #this is an expert demonstration
                    for j in range(expert_actions.shape[1]):
                        if expert_actions[i,j] == 1:
                            if agent_actions[i,j] != 1:
                                #if agent and expert had different predictions, increase l
                                l[i,j] = self.large_margin
                                #and enable supervised loss for this action
                                lam_2[i,j] = self.lam_2
                else:
                    #we revert non-expert transitions back to the action that is stored in the replay buffer.
                    #action choices are typically static in DQNs (off-policy), but DQfD complicates things by comparing the
                    #choices of the agent to its expert demonstrations.
                    agent_actions[i,:] = masks[i,:]

            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, targets_n, importance_weights, agent_actions, l, lam_2, masks], [dummy_targets, targets])

            assert len(pr_idxs) == self.batch_size
            #Calculate new priorities.
            y_true = targets
            #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
            #Boost for expert transitions is handled in memory.PartitionedMemory.update_priorities
            new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + .001
            assert len(new_priorities) == self.batch_size
            self.memory.update_priorities(pr_idxs, new_priorities)

            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def get_config(self):
        config = super(DQNAgent, self).get_config()
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

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))
