from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input
from keras.optimizers import Adam
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import PrioritizedMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.layers import NoisyNetDense

#We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
#a sense of direction and speed.
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

#Standard Atari processing
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='GopherDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(231)
env.seed(123)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

#Standard DQN model architecture, but swapping the Dense classifier layers for the rl.layers.NoisyNetDense version.
input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame = Input(shape=(input_shape))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense= Flatten()(cv3)
dense = NoisyNetDense(512, activation='relu')(dense)
buttons = NoisyNetDense(nb_actions, activation='linear')(dense)
model = Model(inputs=frame,outputs=buttons)
model.summary()

#You can use any Memory you want.
memory = PrioritizedMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=10000000, window_length=WINDOW_LENGTH)

processor = AtariProcessor()

#This is the important difference. Rather than using an E Greedy approach, where
#we keep the network consistent but randomize the way we interpret its predictions,
#in NoisyNet we are adding noise to the network and simply choosing the best value.
policy = GreedyQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., custom_model_objects={"NoisyNetDense":NoisyNetDense})

#Prioritized Memories typically use lower learning rates
lr = .00025
if isinstance(memory, PrioritizedMemory):
    lr /= 4
dqn.compile(Adam(lr=lr), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'noisynet_pdd_dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'noisynet_pdd_dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'noisynet_pdd_dqn_' + args.env_name + '_REWARD_DATA.txt'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)]
    callbacks += [TrainEpisodeLogger(log_filename)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, verbose=0, nb_max_episode_steps=20000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

elif args.mode == 'test':
    weights_filename = 'noisynet_pdd_dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_start_steps=80)
