import numpy as np
from gym import core, spaces
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras import initializations

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class ChainEnvFig3(core.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    shape = [100]

    def __init__(self):
        self.n = 3  # max n = 100
        self.observation_space = spaces.Box(np.zeros(self.shape),np.ones(self.shape))
        self.action_space = spaces.Discrete(2)
        self.viewer = None
        self.obs = np.zeros([100])

    def _reset(self):
        self.t = 1
        self.state = 1
        return self.observe()

    def _step(self, action):
        # action in {0,1}
        self.state = np.clip(self.state - 1 + 2 * action, 0, self.n - 1)
        
        if self.state == 0:
            reward = .001
        elif self.state == self.n - 1:
            reward = 1.
        else:
            reward = 0.
        
        terminal = not self.t < 9 + self.n

        self.t += 1

        obs = self.observe()
        #print obs, action, self.state
        return (obs, reward, terminal, {})

    def observe(self):
        self.obs[:] = 0
        self.obs[:self.state] = 1
        pixels = np.reshape(self.obs, self.shape)
        return pixels



class AlwaysOnDropout(Dropout):
    def __init__(self, p, mode=0, **kwargs):
        super(AlwaysOnDropout, self).__init__(p=p, **kwargs)
        self.mode = mode

    def build(self, input_shape):
        self.stateful = True
        self.mask_shape = input_shape[1:]
        self.stateful_mask = K.ones(self.mask_shape)
        self.is_target_network = K.zeros((1,))
        self.reset_states()

    def reset_states(self):
        K.set_value(self.stateful_mask, np.random.binomial(1, p=self.p, size=self.mask_shape))

    def call(self, x, mask=None):
        noise_shape = self._get_noise_shape(x)
        if self.mode == 0:
            x = x * self.stateful_mask
        elif self.mode == 1:
            x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x * self.stateful_mask)
        elif self.mode == 2:
            x = K.dropout(x, self.p, noise_shape)
        elif self.mode == 3:
            target_mode = K.dropout(x, self.p, noise_shape)
            non_target_mode = K.in_train_phase(K.dropout(x, self.p, noise_shape), x * self.stateful_mask)
            x = K.switch(self.is_target_network[0], target_mode, non_target_mode)
        return x


dropout_mode = 1
chain_length = 50
for _ in xrange(10):
    env = ChainEnvFig3()
    env.n = chain_length
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(AlwaysOnDropout(.2, mode=dropout_mode))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(AlwaysOnDropout(.2, mode=dropout_mode))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(AlwaysOnDropout(.2, mode=dropout_mode))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=50000)
    policy = EpsGreedyQPolicy(eps=0.)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy, enable_double_dqn=False, custom_model_objects={'AlwaysOnDropout': AlwaysOnDropout})
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=2000, visualize=False, verbose=1)
    dqn.test(env, nb_episodes=5, visualize=False)
