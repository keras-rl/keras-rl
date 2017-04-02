import numpy as np
import gym
import argparse
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.keras_future import concatenate, Model


class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


ENV_NAME = 'Pendulum-v0'


def main(argv):
    default_path = 'cdqn_{}_weights.h5f'.format(ENV_NAME)
    default_steps = 50000
    parser = argparse.ArgumentParser(description='Train a CDQN to on Gym environment [{}].'.format(ENV_NAME))
    parser.add_argument('--path', action="store",
                        help='path for saving or loading model [default: {}]'.format(default_path))
    parser.add_argument('--load', action="store_true",
                        help='load pretrained model instead of training new model')
    parser.add_argument('--steps', action="store", type=int, default=default_steps,
                        help='load pretrained model instead of training new model')
    parser.add_argument('--quiet', action='store_true', help='do not visualize training')
    args = parser.parse_args(argv)


    gym.undo_logger_setup()


    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Build all necessary models: V, mu, and L networks.
    V_model = Sequential()
    V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))
    V_model.add(Activation('linear'))
    print(V_model.summary())

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))
    print(mu_model.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = concatenate([action_input, Flatten()(observation_input)])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
    x = Activation('linear')(x)
    L_model = Model(input=[action_input, observation_input], output=x)
    print(L_model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    processor = PendulumProcessor()
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=100, random_process=random_process,
                               gamma=.99, target_model_update=1e-3, processor=processor)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    if args.load:
        agent.load_weights(args.path)
    else:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        agent.fit(env, nb_steps=args.steps, visualize=not args.quiet, verbose=1, nb_max_episode_steps=200)

        # After training is done, we save the final weights.
        agent.save_weights(args.path, overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)

if __name__=="__main__":
    main(sys.argv[1:])
