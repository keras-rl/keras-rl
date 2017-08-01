from __future__ import division
import argparse
import os

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute, LSTM, TimeDistributed, Conv2D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import EpisodicMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def experiment(env_name='Breakout-v0', mode='train', weights=None):
    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # We patch the environment to be closer to what Mnih et al. actually do: The environment
    # repeats the action 4 times and a game is considered to be over during training as soon as a live is lost.
    def _step(a):
        reward = 0.0
        action = env.env._action_set[a]
        # action = env._action_set[a]
        lives_before = env.env.ale.lives()
        # lives_before = env.ale.lives()
        for _ in range(4):
            reward += env.env.ale.act(action)
        ob = env.env._get_obs()
        # ob = env._get_obs()
        done = env.env.ale.game_over() or (mode == 'train' and lives_before != env.env.ale.lives())
        # done = env.ale.game_over() or (args.mode == 'train' and lives_before != env.ale.lives())
        return ob, reward, done, {}
    env._step = _step

    def build_model(stateful, batch_size_selected=None):
        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
        # TODO: fix TF   Jul 20, 2017 confirm change below with Matthias
        if stateful:
            input_shape = (batch_size_selected, None, 1) + INPUT_SHAPE
        else:
            input_shape = (None, 1) + INPUT_SHAPE
        model = Sequential()
        if K.image_dim_ordering() == 'tf':
            # (width, height, channels)
            if stateful:
                model.add(Permute((1, 3, 4, 2), batch_input_shape=input_shape))
                # model.add(Permute((2, 3, 4, 5), batch_input_shape=input_shape))
            else:
                model.add(Permute((1, 3, 4, 2), input_shape=input_shape))
        elif K.image_dim_ordering() == 'th':
            # (channels, width, height)
            if stateful:
                model.add(Permute((1, 2, 3, 4), batch_input_shape=input_shape))
            else:
                model.add(Permute((1, 2, 3, 4), input_shape=input_shape))
        else:
            raise RuntimeError('Unknown image_dim_ordering.')
        model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4))))
        # model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4))))
        model.add(Activation('relu'))
        model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2))))
        # model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2))))
        model.add(Activation('relu'))
        model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1))))
        # model.add(TimeDistributed(Convolution2D(64, 3, 3, subsample=(1, 1))))
        model.add(Activation('relu'))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128, return_sequences=True, stateful=stateful))
        model.add(TimeDistributed(Dense(nb_actions)))
        model.add(Activation('linear'))
        return model

    train_steps = 1750000  # 250001 up to 1750000
    checkpoint_interval = 250000  # default 250000
    train_interval = 100  # default 10
    output_directory = 'output'
    batch_size = 32

    model = build_model(stateful=True, batch_size_selected=batch_size)
    policy_model = build_model(stateful=True, batch_size_selected=1)
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = EpisodicMemory(limit=10000, window_length=1)
    processor = AtariProcessor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, delta_range=(-1., 1.),
                   target_model_update=10000, train_interval=train_interval, policy_model=policy_model,
                   enable_double_dqn=False, batch_size=batch_size)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
        weights_filename = output_directory + '/dqn_{}_weights.h5f'.format(env_name)
        checkpoint_weights_filename = output_directory + '/dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = output_directory + '/dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
        callbacks += [FileLogger(log_filename, interval=100)]
        # dqn.load_weights(weights_filename)
        dqn.fit(env, nb_steps=train_steps, callbacks=callbacks, verbose=1, progbar_show=False, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)
    elif mode == 'test':
        weights_filename = output_directory + '/dqn_{}_weights.h5f'.format(env_name)
        if weights:
            weights_filename = weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='Breakout-v0')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    experiment(args.env_name, args.mode)
