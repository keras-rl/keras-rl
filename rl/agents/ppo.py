import os

import numpy
from keras import optimizers, Input
from keras.engine import Model
from keras.layers import Lambda

from rl.core import Agent
from rl.util import clone_optimizer, clone_model

import keras.backend as K


class PPOAgent(Agent):
    # Note on network architecture:
    # actor should take as input the current state and candidate action, and should output a scalar
    # representing the probability of taking that action.
    # critic's only input is the current state, and should output a scalar representing estimated
    # value of this state
    def __init__(self, actor, critic, memory, epsilon=0.2, nb_actor=3, nb_steps=1000, epoch=5, **kwargs):
        super(Agent, self).__init__(**kwargs)

        # Parameters.
        self.epsilon = epsilon
        self.nb_actor = nb_actor
        self.nb_steps = nb_steps
        self.epoch = epoch

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory

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
        advantage = Input(name='advantage')
        prob_theta = self.actor(action, state)
        prob_thetaold = self.target_actor(action, state)
        def clipped_loss(args):
            prob_theta, prob_thetaold, advantage = args
            prob_ratio = prob_theta / prob_thetaold
            return K.minimum(prob_ratio * advantage, K.clip(prob_ratio, 1-self.epsilon, 1+self.epsilon) * advantage)
        loss_out = Lambda(clipped_loss, name='loss')([prob_theta, prob_thetaold, advantage])
        trainable_model = Model(inputs=[action, state, advantage], outputs=loss_out)
        losses = [ lambda sample_out, network_out: network_out ]
        trainable_model.compile(optimizer=optimizer, loss=losses)
        self.trainable_model = trainable_model


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

    def forward(self, observation):
        # TODO
        prob_dist = self.target_actor.predict_on_batch({ 'state': numpy.repeat(observation, self.nb_action),
                                                         'action': numpy.arange(self.nb_action) })
        return self.policy.select_action(prob_dist)

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        # TODO
