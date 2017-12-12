import os
import itertools
from collections import namedtuple

import numpy as np
from keras import optimizers, Input
from keras.engine import Model
from keras.layers import Lambda

from rl.core import Agent
from rl.memory import FixedBuffer
from rl.util import clone_optimizer, clone_model, GeneralizedAdvantageEstimator

import keras.backend as K

EpisodeMemory = namedtuple('EpisodeMemory', 'state,windowed_state,action,reward,advantage')

def state_windowing(states, window_len):
    def naive_pad(x, shift, axis=0):
        y = np.roll(x, shift, axis)
        y[0:shift, ] = 0
        return y

    return np.stack( [ naive_pad(states, i, axis=0) for i in reversed(range(window_len))], axis=1 )

# TODO: We would need a new memory class that returns windowed info also for rewards
class PPOAgent(Agent):
    """
    Single threaded implementation of the Proximal Policy Optimization algorithm, using an A2C (Advantage
    Actor-Critic) architecture.

    Note on network architecture
    ============================
    actor should take as input a window of recent state, and should output a vector in the abstract
    sample space as defined by the user, which is then feed into the sampler, again supplied
    by the user. This sampler is responsible for transforming the abstract sample space into
    actual value in the action space, through a (known, fixed) random distribution
    actor input shape: (window_length, dimension of observation space)
    critic's only input is the current state, and should output a scalar representing estimated
    value of this state
    critic input shape: (dimension of observation space,)

    :param actor: Actor network
    :param critic: Critic network
    :param memory: Memory object to hold a history of simulation run. As opposed to other agents, we only use it superficially as replay buffer is not used in A2C architecture.
    :param sampler: User supplied sampler. See notes above for explanation.
    :param batch_size: Minibatch size used during training.
    :param epsilon: Cutoff for the clipped loss function
    :param nb_actor: Number of concurrent actors running simulation (They are still executed sequential in this version)
    :param nb_steps: Number of steps to run in each simulation episode before termination. Should match with xx in yy
    :param epoch: Number of training epoch for the actor network.
    :param gamma: Parameter gamma for the GAE.
    :param lamb: Parameter lamb for the GAE.
    """
    def __init__(self, actor, critic, memory, sampler, batch_size=16, epsilon=0.2, nb_actor=3, nb_steps=1000, epoch=5,
                 gamma=0.9, lamb=0.95, **kwargs):
        super(Agent, self).__init__(**kwargs)

        # Parameters.
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.nb_actor = nb_actor
        self.nb_steps = nb_steps
        self.epoch = epoch
        self.gamma = gamma
        self.lamb = lamb

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.sampler = sampler

        # Initialize buffers
        self.episode_memories = [EpisodeMemory(state=FixedBuffer(self.nb_steps), windowed_state=None,
                                               action=FixedBuffer(self.nb_steps), reward=FixedBuffer(self.nb_steps),
                                               advantage=None)
                                 for _ in range(self.nb_actor)]

    def compile(self, optimizer, metrics=[]):
        # TODO
        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        # Compile networks:
        # Critic is used in a standard way, so nothing special.
        # Actor is the ephemeral network that is directly trained,
        # While target_network is the actual actor network and is updated with the weight of actor
        # after each round of training
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_actor.name += '_copy'
        for layer in self.target_actor.layers:
            layer.trainable = False
        #self.actor.compile(optimizer='sgd', loss='mse')
        self.critic.compile(optimizer=critic_optimizer)

        # TODO: Model for the overall objective
        action = Input(name='action')
        state = Input(name='state')
        advantage = Input(name='advantage', shape=(1,))
        actor_out = self.actor([state] + self.actor.inputs[1:])
        target_actor_out = self.target_actor([state] + self.target_actor.inputs[1:])
        log_prob_theta = Lambda(self.sampler.get_dist)(actor_out + [action])
        log_prob_thetaold = Lambda(self.sampler.get_dist)(target_actor_out + [action])
        def clipped_loss(args):
            log_prob_theta, log_prob_thetaold, advantage = args
            prob_ratio = K.exp(log_prob_theta - log_prob_thetaold)
            return K.minimum(prob_ratio * advantage, K.clip(prob_ratio, 1-self.epsilon, 1+self.epsilon) * advantage)
        loss_out = Lambda(clipped_loss, name='loss')([log_prob_theta, log_prob_thetaold, advantage])
        trainable_model = Model(inputs=[action, state, advantage], outputs=loss_out)
        losses = [ lambda sample_out, network_out: network_out ]
        trainable_model.compile(optimizer=optimizer, loss=losses)
        self.trainable_model = trainable_model

        # Other init
        self.round = 0
        self.current_episode_memory = self.episode_memories[0]
        self.finalize_episode = False

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def _complete_episode(self, episode):
        episode_len = len(episode.state) - 1
        assert episode_len == len(episode.action) and episode_len == len(episode.reward)
        windowed_states = state_windowing(np.array(episode.state.get_list()), self.memory.window_length)
        episode.windowed_state = FixedBuffer(self.nb_steps, val=windowed_states)
        # Compute GAE
        gae = GeneralizedAdvantageEstimator(self.critic, windowed_states, np.array(episode.reward.get_list()),
                                            self.gamma, self.lamb)
        episode.advantage = FixedBuffer(self.nb_steps, val=gae)

    def forward(self, observation):
        state_window = self.memory.get_recent_state(observation)
        batch_single = self.process_state_batch([ state_window ])
        actor_out = [ x.flatten() for x in self.target_actor.predict_on_batch(batch_single) ]
        action = self.sampler.sample(actor_out)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    def backward(self, reward, terminal=False):
        if self.finalize_episode:
            self.finalize_episode = False
            # No need to append to the usual memory since next episode will not see anything in last episode anyway?
            self.current_episode_memory.state.append(self.recent_observation)
            self._complete_episode(self.current_episode_memory)

            self.round += 1
            self.current_episode_memory = self.episode_memories[self.round % self.nb_actor]
        else:
            if terminal:
                self.finalize_episode = True
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
            self.current_episode_memory.state.append(self.recent_observation)
            self.current_episode_memory.action.append(self.recent_action)
            self.current_episode_memory.reward.append(reward)
            return

        if (self.round % self.nb_actor == 0) and self.round > 0:
            # do training
            batch_state     = np.concatenate([ self.episode_memories[i].windowed_state.get_list() for i in range(self.nb_actor) ])
            batch_reward    = np.concatenate([ self.episode_memories[i].reward.get_list()         for i in range(self.nb_actor) ])
            batch_advantage = np.concatenate([ self.episode_memories[i].advantage.get_list()      for i in range(self.nb_actor) ])
            batch_action    = np.concatenate([ self.episode_memories[i].action.get_list()         for i in range(self.nb_actor) ])

            # Set dummy output with matching shape
            assert batch_state.shape[0] == batch_reward.shape[0] and batch_state.shape[0] == batch_advantage.shape[0] \
                and batch_state.shape[0] == batch_action.shape[0]
            total_batch_size = batch_state.shape[0]
            dummy_out = np.zeros((total_batch_size, 1))
            self.trainable_model.fit([batch_action, batch_state, batch_advantage],
                                     dummy_out, epochs=self.epoch, batch_size=self.batch_size)
            # Copy results to target_actor
            self.target_actor.set_weights(self.actor.get_weights())

            #TODO: train value network
            predict_value = self.critic.predict_on_batch(batch_state[1:]).flatten()
            # Compute r_t + gamma * V(s_t+1) and update the target ys accordingly,
            discounted_reward_batch = self.gamma * predict_value
            #discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == batch_reward.shape
            targets = (batch_reward + discounted_reward_batch).reshape(self.batch_size, 1)

            self.critic.train_on_batch(batch_state[:-1], targets)

            # reset all episode memories
            self.episode_memories = [EpisodeMemory(state=FixedBuffer(self.nb_steps), windowed_state=None,
                                                   action=FixedBuffer(self.nb_steps), reward=FixedBuffer(self.nb_steps),
                                                   advantage=None)
                                     for _ in range(self.nb_actor)]
            self.current_episode_memory = self.episode_memories[0]
