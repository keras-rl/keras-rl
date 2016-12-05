from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose

from rl.memory import SequentialMemory
from rl.core import Agent, Env, Processor


class TestEnv(Env):
    def __init__(self):
        super(TestEnv, self).__init__()

    def step(self, action):
        self.state += 1
        done = self.state >= 6
        reward = float(self.state) / 10.
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = 1
        return np.array(self.state)

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass


class TestAgent(Agent):
    def __init__(self, memory, **kwargs):
        super(TestAgent, self).__init__(**kwargs)
        self.memory = memory

    def forward(self, observation):
        action = observation
        self.recent_action = action
        self.recent_observation = observation
        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in self.metrics_names]
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal)
        return metrics

    def compile(self):
        self.compiled = True


def test_fit_observations():
    memory = SequentialMemory(100, window_length=2, ignore_episode_boundaries=False)
    agent = TestAgent(memory)
    env = TestEnv()
    agent.compile()
    agent.fit(env, 20, verbose=0)

    # Inspect memory to see if observations are correct.
    experiencies = memory.sample(batch_size=8, batch_idxs=range(8))
    
    assert experiencies[0].reward == .2
    assert experiencies[0].action == 1
    assert_allclose(experiencies[0].state0, np.array([0, 1]))
    assert_allclose(experiencies[0].state1, np.array([1, 2]))
    assert experiencies[0].terminal1 is False
    
    assert experiencies[1].reward == .3
    assert experiencies[1].action == 2
    assert_allclose(experiencies[1].state0, np.array([1, 2]))
    assert_allclose(experiencies[1].state1, np.array([2, 3]))
    assert experiencies[1].terminal1 is False

    assert experiencies[2].reward == .4
    assert experiencies[2].action == 3
    assert_allclose(experiencies[2].state0, np.array([2, 3]))
    assert_allclose(experiencies[2].state1, np.array([3, 4]))
    assert experiencies[2].terminal1 is False

    assert experiencies[3].reward == .5
    assert experiencies[3].action == 4
    assert_allclose(experiencies[3].state0, np.array([3, 4]))
    assert_allclose(experiencies[3].state1, np.array([4, 5]))
    assert experiencies[3].terminal1 is False

    assert experiencies[4].reward == .6
    assert experiencies[4].action == 5
    assert_allclose(experiencies[4].state0, np.array([4, 5]))
    assert_allclose(experiencies[4].state1, np.array([5, 6]))
    assert experiencies[4].terminal1 is True

    # Experience 5 has been re-sampled since since state0 would be terminal in which case we
    # cannot really have a meaningful transition because the environment gets reset. We thus
    # just ensure that state0 is not terminal.
    assert not np.all(experiencies[5].state0 == np.array([5, 6]))

    assert experiencies[6].reward == .2
    assert experiencies[6].action == 1
    assert_allclose(experiencies[6].state0, np.array([0, 1]))
    assert_allclose(experiencies[6].state1, np.array([1, 2]))
    assert experiencies[6].terminal1 is False

    assert experiencies[7].reward == .3
    assert experiencies[7].action == 2
    assert_allclose(experiencies[7].state0, np.array([1, 2]))
    assert_allclose(experiencies[7].state1, np.array([2, 3]))
    assert experiencies[7].terminal1 is False


def test_copy_observations():
    methods = [
        'fit',
        'test',
    ]

    for method in methods:
        original_observations = []
        
        class LocalEnv(Env):
            def __init__(self):
                super(LocalEnv, self).__init__()

            def step(self, action):
                self.state += 1
                done = self.state >= 6
                reward = float(self.state) / 10.
                obs = np.array(self.state)
                original_observations.append(obs)
                return obs, reward, done, {}

            def reset(self):
                self.state = 1
                return np.array(self.state)

            def seed(self, seed=None):
                pass

            def configure(self, *args, **kwargs):
                pass

        # Slight abuse of the processor for test purposes.
        observations = []

        class LocalProcessor(Processor):
            def process_step(self, observation, reward, done, info):
                observations.append(observation)
                return observation, reward, done, info

        processor = LocalProcessor()
        memory = SequentialMemory(100, window_length=1)
        agent = TestAgent(memory, processor=processor)
        env = LocalEnv()
        agent.compile()
        getattr(agent, method)(env, 20, verbose=0, visualize=False)

        assert len(observations) == len(original_observations)
        assert_allclose(np.array(observations), np.array(original_observations))
        assert np.all([o is not o_ for o, o_ in zip(original_observations, observations)])
    

if __name__ == '__main__':
    pytest.main([__file__])
