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


class PPOAgent(Agent):
    """
    Single threaded implementation of the Proximal Policy Optimization algorithm, using an A2C (Advantage
    Actor-Critic) architecture.

    Note on network architecture
    ============================
    The actor network should take as input a window of recent states, and should output a vector in the abstract
    sample space as defined by the user, which is then feed into the sampler, again supplied
    by the user. This sampler (instance of interface ``ProbabilityDistribution``) is responsible for
    transforming the abstract sample space into actual value in the action space, through a (known,
    fixed) random distribution, by calling ``self.sampler.sample``.
    The actor network can have dummy inputs. An example use case would be to act as the source of
    trainable parameter independent from the network, such as the noise level/volatility to add to output.
    In such case all but one of the multiple inputs of the network must be dummy, and the index of the
    actual (sole) input (as occur in ``network.inputs``) should be specified through ``actor_input_index``.
    The critic network have the same input as actor, and should output a scalar representing estimated
    value of this state.

    Algorithm
    =========
    TODO

    Internal design
    ===============
    First, we replaced the usual ``Memory`` class with our own ``EpisodeMemory`` to deal with the unique
    demand in this scenario. In particular, we need to deal with windowing, and more importantly the
    custom objective function contains a value (Advantage Estimate) that is derived from the state transitions
    in a rather non-trivial way - this is handled by leaving ``windowed_state`` and ``advantage`` blank until
    the very end of the episode, upon which we do all the computations (which information-theoretically do
    depends on the entire episode anyway), to recover all values and fill it in via ``_complete_episode``.
    Another intrinsic difference is that in the basic framework any individual datum in the experience
    replay buffer is disposable, which is not true in A2C since we perform training in batch here. (Aside
    from the need to retain data to derive the missing info) Hence we also use our own buffer implementation
    called ``FixedBuffer``. This works because PPO uses the GAE which has a fixed size lookahead that is known
    statically.

    As for the A2C framework itself, we use the attribute ``self.episode_memories`` to hold the memories/buffers -
    one for each concurrent actor, so it is a python list of ``EpisodeMemory``. Since this is a single-threaded
    implementation, we will still adhere to the framework of the base class ``Agent``, and only simulate the
    concurrency via round-robin execution. To be precise, as we simulate many runs using the actor network,
    the information of each episode will be saved to each concurrent actor one by one. One batch mode training
    will occur each time a full round is completed - that is, when each concurrent actor has saved one new episode.
    When this happen we aggregate the (new) episodes from each actors into a big, fixed data set, which is then
    feed to the training algorithm for some iterations. (this ensure information efficiency by making sure the
    learning value in those episodes are more fully exploited) After that, both networks are updated, the
    data sets and all buffers are cleared, and we begin anew.

    :param actor: Actor network, input shape ``(window_length, dimensions of observation space)``. (Keras' own batch dimension is implicit as usual)
    :param actor_input_index: Index of the actual/non-dummy input to the actor network as occur in ``network.inputs``. Ignored if there is only one input.
    :param critic: Critic network, input shape ``(window_length, dimensions of observation space)``. (Keras' own batch dimension is implicit as usual)
    :param memory: ``Memory`` object to hold a history of simulation run. As opposed to other agents, we only use it superficially as replay buffer is not used in A2C architecture.
    :param sampler: User supplied sampler, should be an instance of the interface ``ProbabilityDistribution``. See notes above for explanation of its role.
    :param batch_size: Minibatch size used during training.
    :param epsilon: Cutoff for the clipped loss function
    :param nb_actor: Number of concurrent actors running simulation (They are still executed sequential in this version)
    :param nb_steps: Number of steps to run in each simulation episode before termination. Should match with xx in yy
    :param epoch: Number of training epoch for the actor network.
    :param gamma: Parameter gamma for the GAE.
    :param lamb: Parameter lamb for the GAE.
    """
    def __init__(self, actor, actor_input_index, critic, memory, sampler, batch_size=16, epsilon=0.2, nb_actor=3, nb_steps=1000, epoch=5,
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
        self.actor_input_index = actor_input_index

        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. PPO expects a critic that has a single output.'.format(critic))
        if hasattr(critic.inputs, '__len__') and len(critic.inputs) > 1:
            raise ValueError('Critic "{}" has more than one input. PPO expects a critic that has a single input.'.format(critic))
        # Validate actor_input_index
        if not hasattr(actor.inputs, '__len__') or len(actor.inputs) == 1:
            self.actor_input_index = 0
        elif actor_input_index < 0 or actor_input_index >= len(actor.inputs):
            raise ValueError('actor_input_index out of the range of actor "{}"\'s input list.'.format(actor))

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

    def _get_replaced_actor_input_list(self, state_input, target=False):
        s = self.actor_input_index
        if target:
            network = self.target_actor
        else:
            network = self.actor
        return network.inputs[0:s] + [state_input] + network.inputs[s+1:]

    def _get_actor_dummy_inputs(self, target=False):
        s = self.actor_input_index
        if target:
            network = self.target_actor
        else:
            network = self.actor
        return network.inputs[0:s] + network.inputs[s + 1:]

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
            layer.name += '_copy'
        #self.actor.compile(optimizer='sgd', loss='mse')
        self.critic.compile(optimizer=critic_optimizer)

        # TODO: Model for the overall objective
        action = Input(name='action')
        state = Input(name='state')
        advantage = Input(name='advantage', shape=(1,))
        actor_out = self.actor(self._get_replaced_actor_input_list(state))
        target_actor_out = self.target_actor(self._get_replaced_actor_input_list(state, target=True))
        log_prob_theta = Lambda(self.sampler.get_dist)(actor_out + [action])
        log_prob_thetaold = Lambda(self.sampler.get_dist)(target_actor_out + [action])
        def clipped_loss(args):
            log_prob_theta, log_prob_thetaold, advantage = args
            prob_ratio = K.exp(log_prob_theta - log_prob_thetaold)
            return K.minimum(prob_ratio * advantage, K.clip(prob_ratio, 1-self.epsilon, 1+self.epsilon) * advantage)
        loss_out = Lambda(clipped_loss, name='loss')([log_prob_theta, log_prob_thetaold, advantage])
        trainable_model = Model(inputs=[action, state, advantage] +
                                       self._get_actor_dummy_inputs(target=True) +
                                       self._get_actor_dummy_inputs(target=False), 
                                outputs=loss_out)
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
