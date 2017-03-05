from rl.core import Agent
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.agents.dqn import mean_q
from rl.util import huber_loss
from keras.layers import Input, Lambda
from keras.models import Model
import keras.backend as K
from rl.policy import EpsGreedyQPolicy
from rl.util import get_object_config


class Sarsa(Agent):
    def __init__(self, model, nb_actions, policy=EpsGreedyQPolicy(), gamma=.99, nb_steps_warmup=10,
                 train_interval=1, delta_range=None, delta_clip=np.inf, *args, **kwargs):
        super(Sarsa, self).__init__(*args, **kwargs)

        self.state0 = None
        self.action0 = None
        self.next_action = None
        self.model = model
        self.nb_actions = nb_actions
        self.policy = policy
        self.gamma = gamma
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        self.delta_clip = delta_clip
        self.compiled = False

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def get_config(self):
        config = super(Sarsa, self).get_config()
        config['nb_actions'] = self.nb_actions
        config['gamma'] = self.gamma
        config['nb_steps_warmup'] = self.nb_steps_warmup
        config['train_interval'] = self.train_interval
        config['delta_clip'] = self.delta_clip
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

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
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
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

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        if self.compiled:
            self.model.reset_states()

    def clean_on_eposide_end(self):
        self.next_action = None

    def forward(self, observation):
        # on policy algorithms follow the next action calculated in update formula in backward method
        if self.next_action is None or self.training is False:
            q_values = self.compute_q_values([observation])
            action = self.policy.select_action(q_values=q_values)
            if self.processor is not None:
                action = self.processor.process_action(action)

            # SARSA needs to know (State0, Action0)
            self.state0 = observation
            self.action0 = action

            return action
        else:
            # SARSA needs to know (State0, Action0)
            self.state0 = observation
            self.action0 = self.next_action
            return self.next_action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = [self.state0]
            reward_batch = [reward]
            action_batch = [self.action0]
            terminal1_batch = [0.] if terminal else [1.]
            state1_batch = [self.observation]

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (1,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            batch = self.process_state_batch(state1_batch)
            # assert batch is not None
            # print batch
            q_values = self.compute_q_values(batch)
            q_values = q_values.reshape((1, self.nb_actions))

            action = [self.policy.select_action(q_values=q_values[0, :])]
            if self.processor is not None:
                action = self.processor.process_action(action)

            self.next_action = action[0]
            q_batch = q_values[0, action]

            assert q_batch.shape == (1,)
            targets = np.zeros((1, self.nb_actions))
            dummy_targets = np.zeros((1,))
            masks = np.zeros((1, self.nb_actions))

            # Compute r_t + gamma * Q(s_t+1, a_t+1)
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
            state0_batch = state0_batch.reshape((1,) + state0_batch.shape)
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics
        return metrics