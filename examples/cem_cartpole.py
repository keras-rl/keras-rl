import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory

class GaussianModel(object):
    def __init__(self,nb_actions=1,obs_dim=1):
        self.nb_actions = nb_actions
        self.obs_dim = obs_dim
        self.W = np.random.randn(obs_dim, nb_actions)
        self.b = np.random.randn(1, nb_actions)

    def update_params(self,theta):
        assert theta.shape == ((self.obs_dim + 1) * self.nb_actions,)
        self.W = theta[0 : self.obs_dim * self.nb_actions].reshape(self.obs_dim, self.nb_actions)
        self.b = theta[self.obs_dim * self.nb_actions : None].reshape(1, self.nb_actions)

    def get_params(self):
        theta = np.zeros((self.obs_dim+1)*self.nb_actions)
        theta[0 : self.obs_dim * self.nb_actions] = self.W.reshape((1,self.obs_dim*self.nb_actions))
        theta[self.obs_dim*self.nb_actions:None] = self.b
        return theta

    def select_action(self,observation):
        y = observation.dot(self.W) + np.hstack([self.b]*observation.shape[0])
        a = y.argmax()
        return a

    @property
    def metrics_names(self):
        return ['mse']


ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

model = GaussianModel(nb_actions,obs_dim)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
#memory = SequentialMemoryWithParams(limit=50000)
memory = EpisodeParameterMemory(limit=1000)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10)
cem.compile(Adam(lr=1e-3),metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=20000, visualize=True, verbose=2)

# After training is done, we save the final weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)
