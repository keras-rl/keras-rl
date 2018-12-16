import os
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'MountainCar-v0'
env = gym.make(ENV_NAME)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


memory = SequentialMemory(limit=50000, window_length=1)


policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)


dqn.compile(Adam(lr=1e-3), metrics=['mae'])


model_path = 'dqn_{}_model.h5f'.format(ENV_NAME)

if os.path.exists(model_path):
	print("resume training")
	dqn.load(model_path)

dqn.fit(env, nb_steps=50000, action_repetition = 5, visualize=False, verbose=1)
dqn.save(model_path, overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)