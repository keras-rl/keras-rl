import os

import numpy as np
import keras.backend as K

from rl.core import Agent
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DiscreteA3CAgent(Agent):
    pass


class ContinuousA3CAgent(Agent):
    def __init__(self, nb_actions, actor, critic, actor_mean_output, actor_variance_output,
                 gamma=.9, beta=1e-4, target_model_update=1., reward_range=(-np.inf, np.inf),
                 delta_range=(-np.inf, np.inf), processor=None, custom_objects={},
                 batch_size=32, enable_bootstrapping=True):
        super(ContinuousA3CAgent, self).__init__()

        self.actor = actor
        self.critic = critic
        self.actor_mean_output_idx = self.actor.output.index(actor_mean_output)
        self.actor_variance_output_idx = self.actor.output.index(actor_variance_output)
        self.custom_objects = custom_objects

        # Parameters.
        self.nb_actions = nb_actions
        self.enable_bootstrapping = enable_bootstrapping
        self.target_model_update = target_model_update
        self.beta = beta
        self.gamma = gamma
        self.reward_range = reward_range
        self.delta_range = delta_range
        self.processor = processor
        self.batch_size = batch_size

        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        if hasattr(optimizer, '__len__'):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        assert actor_optimizer != critic_optimizer

        metrics += [mean_q]

        self.local_actor = clone_model(self.actor, self.custom_objects)
        self.local_actor.compile(optimizer='sgd', loss='mse')  # never used for optimization
        self.local_actor._make_predict_function()
        self.local_critic = clone_model(self.critic, self.custom_objects)
        self.local_critic.compile(optimizer='sgd', loss='mse')  # never used for optimization
        self.local_critic._make_predict_function()

        #TODO: check if critic and actor are already compiled
        # TODO: this probably doesn't work for multiple instances
        # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
        critic_updates = get_soft_target_model_updates(self.local_critic, self.critic, self.target_model_update)
        critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        def clipped_mse(y_true, y_pred):
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            return K.mean(K.square(delta), axis=-1)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_mse, metrics=metrics)
        self.critic._make_predict_function()
        self.critic._make_train_function()
        
        # Compile the gradient policy function.
        Vs = K.placeholder(shape=(self.batch_size,))
        Rs = K.placeholder(shape=(self.batch_size,))
        actions = K.placeholder(shape=(self.batch_size, self.nb_actions))
        outputs = self.actor.output
        assert len(outputs) == 2
        means = outputs[self.actor_mean_output_idx]
        variances = outputs[self.actor_variance_output_idx] + K.epsilon()
        
        # Compute the probability of the action under a normal distribution.
        pdf = 1. / K.sqrt(2. * np.pi * variances) * K.exp(-K.square(actions - means) / (2. * variances))
        log_pdf = K.log(pdf + K.epsilon())
        grads = None
        if K._BACKEND == 'tensorflow':
            grads = None
            for idx in xrange(self.batch_size):
                gs = [g * (Rs[idx] - Vs[idx]) for g in K.gradients(log_pdf[idx, :], self.actor.trainable_weights)]
                if grads is None:
                    grads = gs
                else:
                    grads = [g + g_ for g, g_ in zip(gs, grads)]
            grads = [g / float(self.batch_size) for g in grads]
        elif K._BACKEND == 'theano':
            import theano.tensor as T
            grads = None
            for idx in xrange(self.batch_size):
                gs = [g.flatten() * (Rs[idx] - Vs[idx]) for g in T.jacobian(log_pdf[idx, :], self.actor.trainable_weights)]
                if grads is None:
                    grads = gs
                else:
                    grads = [g + g_ for g, g_ in zip(gs, grads)]
            grads = [g / float(self.batch_size) for g in grads]
        else:
            raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
        assert grads is not None

        # Compute gradient for the regularization term. We have to add the means here since otherwise
        # computing the gradient fails due to an unconnected graph.
        regularizer = -.5 * (K.log(2. * np.pi * variances) + 1) + 0. * means
        regularizer_grads = None
        if K._BACKEND == 'tensorflow':
            regularizer_grads = K.gradients(regularizer, self.actor.trainable_weights)
        elif K._BACKEND == 'theano':
            import theano.tensor as T
            regularizer_grads = T.jacobian(regularizer.flatten(), self.actor.trainable_weights)
            regularizer_grads = [K.mean(g, axis=0) for g in regularizer_grads]
        else:
            raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))
        assert regularizer_grads is not None
        assert len(grads) == len(regularizer_grads)
        
        # Combine grads.
        grads = [g + self.beta * rg for g, rg in zip(grads, regularizer_grads)]

        # We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
        # the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
        clipnorm = getattr(actor_optimizer, 'clipnorm', 0.)
        clipvalue = getattr(actor_optimizer, 'clipvalue', 0.)
        def get_gradients(loss, params):
            # We want to follow the gradient, but the optimizer goes in the opposite direction to
            # minimize loss. Hence the double inversion.
            assert len(grads) == len(params)
            modified_grads = [-g for g in grads]
            if clipnorm > 0.:
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in modified_grads]))
                modified_grads = [optimizers.clip_norm(g, clipnorm, norm) for g in modified_grads]
            if clipvalue > 0.:
                modified_grads = [K.clip(g, -clipvalue, clipvalue) for g in modified_grads]
            return modified_grads
        actor_optimizer.get_gradients = get_gradients
        updates = actor_optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints, None)
        updates += get_soft_target_model_updates(self.local_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN
        # TODO: does this include state updates for RNNs?

        # Finally, combine it all into a callable function.
        actor_inputs = None
        if not hasattr(self.actor.input, '__len__'):
            actor_inputs = [self.actor.input]
        else:
            actor_inputs = self.actor.input
        inputs = actor_inputs + [Vs, Rs, actions]
        if self.uses_learning_phase:
            inputs += [K.learning_phase()]
        assert len(self.actor.output) > 1
        self.actor_train_fn = K.function(inputs, self.actor.output, updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, observation):
        batch = self.process_state_batch([observation])
        means, variances = [x.flatten() for x in self.local_actor.predict_on_batch(batch)]
        assert means.shape == (self.nb_actions,)
        assert variances.shape == (self.nb_actions,)
        action = np.random.normal(means, np.sqrt(variances) + K.epsilon(), size=(self.nb_actions,))
        assert action.shape == (self.nb_actions,)
        return action

    def reset_states(self):
        if getattr(self, 'local_actor', None) is not None:
            self.local_actor.reset_states()
        if getattr(self, 'local_critic', None) is not None:
            self.local_critic.reset_states()

        self.reward_accumulator = []
        self.observation_accumulator = []
        self.action_accumulator = []
        self.terminal_accumulator = []

    @property
    def metrics_names(self):
        return self.critic.metrics_names[:] + ['mean_action', 'mean_action_variance']

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)
        action = self.select_action(observation)
        self.observation_accumulator.append(observation)
        self.action_accumulator.append(action)
        return action

    def backward(self, reward, terminal=False):
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            return metrics

        # Clip the reward to be in reward_range and perform book-keeping.
        reward = min(max(reward, self.reward_range[0]), self.reward_range[1])
        self.reward_accumulator.append(reward)
        self.terminal_accumulator.append(terminal)
        assert len(self.reward_accumulator) == len(self.observation_accumulator)
        assert len(self.reward_accumulator) == len(self.terminal_accumulator)
        assert len(self.reward_accumulator) == len(self.action_accumulator)
        
        perform_update = self.training and len(self.observation_accumulator) > self.batch_size
        if not perform_update:
            # Nothing to do yet, just keep going.
            return metrics

        # We have one more data point to bootstrap from.
        assert len(self.observation_accumulator) == self.batch_size + 1

        # Accumulate data for gradient computation.
        observations = self.process_state_batch(self.observation_accumulator)
        # TODO: make bootstrapping compatbile with LSTMs
        Vs = self.local_critic.predict_on_batch(observations).flatten().tolist()
        if self.enable_bootstrapping:
            R = 0. if self.terminal_accumulator[-1] else Vs[-1]
        else:
            R = 0.
        Rs = [R]
        for r, t in zip(reversed(self.reward_accumulator[:-1]), reversed(self.terminal_accumulator[:-1])):
            R = r + self.gamma * R if not t else r
            Rs.append(R)
        Rs = list(reversed(Rs))

        # Remove latest value, which we have no use for.
        observations = np.array(observations[:-1])
        actions = np.array(self.action_accumulator[:-1])
        rewards = np.array(self.reward_accumulator[:-1])
        terminals = np.array(self.terminal_accumulator[:-1])
        Rs = np.array(Rs[:-1])
        Vs = np.array(Vs[:-1])

        # Ensure that everything is fine and enqueue for update.
        assert observations.shape[0] == self.batch_size
        assert Rs.shape == (self.batch_size,)
        assert Vs.shape == (self.batch_size,)
        assert rewards.shape == (self.batch_size,)
        assert actions.shape == (self.batch_size, self.nb_actions)
        assert terminals.shape == (self.batch_size,)

        # Update critic. This also updates the local critic in the process.
        metrics = self.critic.train_on_batch(observations, Rs)

        # Update the actor.
        inputs = [observations, Rs, Vs, actions]
        if self.uses_learning_phase:
            inputs += [self.training]
        means, variances = self.actor_train_fn(inputs)
        assert means.shape == (self.batch_size, self.nb_actions)
        assert variances.shape == (self.batch_size, self.nb_actions)
        metrics += [np.mean(means), np.mean(variances)]
        
        # Reset state for next update round. We keep the latest data point around since we haven't
        # used it.
        self.observation_accumulator = [self.observation_accumulator[-1]]
        self.action_accumulator = [self.action_accumulator[-1]]
        self.terminal_accumulator = [self.terminal_accumulator[-1]]
        self.reward_accumulator = [self.reward_accumulator[-1]]

        return metrics
