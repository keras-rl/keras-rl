import numpy as np
import gym
import threading

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Nadam

from rl.agents import ContinuousA3CAgent
from rl.callbacks import TrainIntervalLogger


ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor_input = Input(shape=env.observation_space.shape)
x = Dense(32)(actor_input)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
actor_mean_output = Dense(nb_actions)(x)
actor_mean_output = Activation('linear')(actor_mean_output)
actor_variance_output = Dense(nb_actions)(x)
actor_variance_output = Activation('softplus')(actor_variance_output)
actor = Model(input=actor_input, output=[actor_mean_output, actor_variance_output])
print(actor.summary())

critic_input = Input(shape=env.observation_space.shape)
x = Dense(32)(critic_input)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=critic_input, output=x)
print(critic.summary())

agent = ContinuousA3CAgent(nb_actions=nb_actions, actor=actor, critic=critic,
	actor_mean_output=actor_mean_output, actor_variance_output=actor_variance_output,
	gamma=.99, batch_size=5)
agent.compile([Nadam(lr=.001), Nadam(lr=.001)], metrics=['mae'])
agent.fit(gym.make(ENV_NAME), nb_steps=1000000, visualize=True, nb_max_episode_steps=200)

# After training is done, we save the final weights.
agent.save_weights('a3c_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
