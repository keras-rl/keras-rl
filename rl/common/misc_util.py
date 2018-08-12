import gym
import numpy as np
import keras.backend as K
import random

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)