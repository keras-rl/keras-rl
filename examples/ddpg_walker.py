import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, BatchNormalization
from keras.optimizers import Nadam, RMSprop

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


ENV_NAME = 'BipedalWalker-v2'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(BatchNormalization(mode=2))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(BatchNormalization(mode=2))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(BatchNormalization(mode=2))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(BatchNormalization(mode=2))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = BatchNormalization(mode=2)(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = BatchNormalization(mode=2)(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = BatchNormalization(mode=2)(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = BatchNormalization(mode=2)(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
	memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, random_process=random_process,
	gamma=.99, target_model_update=1e-3)
agent.compile([RMSprop(lr=.001), RMSprop(lr=.001)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=20000000, visualize=False, verbose=1, nb_max_episode_steps=1000)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True)
