# Inspired from OpenAI Baselines

import gym
import numpy as np
import random

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
