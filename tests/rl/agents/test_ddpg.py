from __future__ import division
from __future__ import absolute_import

from keras.models import Model, Sequential
from keras.layers import Input, merge, Dense, Flatten

from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor
from tests.rl import util


def test_single_ddpg_input():
    nb_actions = 2

    actor = Sequential()
    actor.add(Flatten(input_shape=(2, 3)))
    actor.add(Dense(nb_actions))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(2, 3), name='observation_input')
    x = merge([action_input, Flatten()(observation_input)], mode='concat')
    x = Dense(1)(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=10, window_length=2)
    agent = DDPGAgent(actor=actor, critic=critic, critic_action_input=action_input, memory=memory,
                      nb_actions=2, nb_steps_warmup_critic=5, nb_steps_warmup_actor=5, batch_size=4)
    agent.compile('sgd')
    # import ipdb; ipdb.set_trace()
    env = util.MultiInputTestEnv((3,))
    agent.fit(env, nb_steps=10)

    # ============================ test_multi_ddpg_input ============================
    # running parallely crashes them both. Te recreate un comment the following line
    # def test_multi_ddpg_input(). Plese discuss this in an issue thread if any one can
    # figure out why
    nb_actions = 2

    actor_observation_input1 = Input(shape=(2, 3), name='actor_observation_input1')
    actor_observation_input2 = Input(shape=(2, 4), name='actor_observation_input2')
    # actor = Sequential()
    x = merge([actor_observation_input1, actor_observation_input2], mode='concat')
    x = Flatten()(x)
    x = Dense(nb_actions)(x)
    actor = Model(input=[actor_observation_input1, actor_observation_input2], output=x)

    action_input = Input(shape=(nb_actions,), name='action_input')
    critic_observation_input1 = Input(shape=(2, 3), name='critic_observation_input1')
    critic_observation_input2 = Input(shape=(2, 4), name='critic_observation_input2')
    x = merge([critic_observation_input1, critic_observation_input2], mode='concat')
    x = merge([action_input, Flatten()(x)], mode='concat')
    x = Dense(1)(x)
    critic = Model(input=[action_input, critic_observation_input1, critic_observation_input2], output=x)

    processor = MultiInputProcessor(nb_inputs=2)
    memory = SequentialMemory(limit=10, window_length=2)
    agent = DDPGAgent(actor=actor, critic=critic, critic_action_input=action_input, memory=memory,
                      nb_actions=2, nb_steps_warmup_critic=5, nb_steps_warmup_actor=5, batch_size=4,
                      processor=processor)
    agent.compile('sgd')
    env = util.MultiInputTestEnv([(3,), (4,)])
    agent.fit(env, nb_steps=10)
