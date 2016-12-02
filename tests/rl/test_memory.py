from __future__ import division
import pytest
import numpy as np
from numpy.testing import assert_allclose

from rl.memory import SequentialMemory, RingBuffer


def test_ring_buffer():
    def assert_elements(b, ref):
        assert len(b) == len(ref)
        for idx in range(b.maxlen):
            if idx >= len(ref):
                with pytest.raises(KeyError):
                    b[idx]
            else:
                assert b[idx] == ref[idx]

    b = RingBuffer(5)

    # Fill buffer.
    assert_elements(b, [])
    b.append(1)
    assert_elements(b, [1])
    b.append(2)
    assert_elements(b, [1, 2])
    b.append(3)
    assert_elements(b, [1, 2, 3])
    b.append(4)
    assert_elements(b, [1, 2, 3, 4])
    b.append(5)
    assert_elements(b, [1, 2, 3, 4, 5])

    # Add couple more items with buffer at limit.
    b.append(6)
    assert_elements(b, [2, 3, 4, 5, 6])
    b.append(7)
    assert_elements(b, [3, 4, 5, 6, 7])
    b.append(8)
    assert_elements(b, [4, 5, 6, 7, 8])


def test_get_recent_state_with_episode_boundaries():
    memory = SequentialMemory(3, window_length=2, ignore_episode_boundaries=False)
    obs_size = (3, 4)
    
    obs0 = np.random.random(obs_size)
    terminal0 = False

    obs1 = np.random.random(obs_size)
    terminal1 = False

    obs2 = np.random.random(obs_size)
    terminal2 = False

    obs3 = np.random.random(obs_size)
    terminal3 = True

    obs4 = np.random.random(obs_size)
    terminal4 = False

    obs5 = np.random.random(obs_size)
    terminal5 = True

    obs6 = np.random.random(obs_size)
    terminal6 = False

    state = np.array(memory.get_recent_state(obs0))
    assert state.shape == (2,) + obs_size
    assert np.allclose(state[0], 0.)
    assert np.all(state[1] == obs0)

    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, 0, 0., terminal1)
    state = np.array(memory.get_recent_state(obs1))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs0)
    assert np.all(state[1] == obs1)

    memory.append(obs1, 0, 0., terminal2)
    state = np.array(memory.get_recent_state(obs2))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs1)
    assert np.all(state[1] == obs2)

    memory.append(obs2, 0, 0., terminal3)
    state = np.array(memory.get_recent_state(obs3))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs2)
    assert np.all(state[1] == obs3)

    memory.append(obs3, 0, 0., terminal4)
    state = np.array(memory.get_recent_state(obs4))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == np.zeros(obs_size))
    assert np.all(state[1] == obs4)

    memory.append(obs4, 0, 0., terminal5)
    state = np.array(memory.get_recent_state(obs5))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs4)
    assert np.all(state[1] == obs5)

    memory.append(obs5, 0, 0., terminal6)
    state = np.array(memory.get_recent_state(obs6))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == np.zeros(obs_size))
    assert np.all(state[1] == obs6)


def test_training_flag():
    obs_size = (3, 4)

    obs0 = np.random.random(obs_size)
    terminal0 = False

    obs1 = np.random.random(obs_size)
    terminal1 = True

    obs2 = np.random.random(obs_size)
    terminal2 = False

    for training in (True, False):
        memory = SequentialMemory(3, window_length=2)

        state = np.array(memory.get_recent_state(obs0))
        assert state.shape == (2,) + obs_size
        assert np.allclose(state[0], 0.)
        assert np.all(state[1] == obs0)
        assert memory.nb_entries == 0
        
        memory.append(obs0, 0, 0., terminal1, training=training)
        state = np.array(memory.get_recent_state(obs1))
        assert state.shape == (2,) + obs_size
        assert np.all(state[0] == obs0)
        assert np.all(state[1] == obs1)
        if training:
            assert memory.nb_entries == 1
        else:
            assert memory.nb_entries == 0

        memory.append(obs1, 0, 0., terminal2, training=training)
        state = np.array(memory.get_recent_state(obs2))
        assert state.shape == (2,) + obs_size
        assert np.allclose(state[0], 0.)
        assert np.all(state[1] == obs2)
        if training:
            assert memory.nb_entries == 2
        else:
            assert memory.nb_entries == 0


def test_get_recent_state_without_episode_boundaries():
    memory = SequentialMemory(3, window_length=2, ignore_episode_boundaries=True)
    obs_size = (3, 4)
    
    obs0 = np.random.random(obs_size)
    terminal0 = False
    
    obs1 = np.random.random(obs_size)
    terminal1 = False
    
    obs2 = np.random.random(obs_size)
    terminal2 = False
    
    obs3 = np.random.random(obs_size)
    terminal3 = True

    obs4 = np.random.random(obs_size)
    terminal4 = False

    obs5 = np.random.random(obs_size)
    terminal5 = True

    obs6 = np.random.random(obs_size)
    terminal6 = False
    
    state = np.array(memory.get_recent_state(obs0))
    assert state.shape == (2,) + obs_size
    assert np.allclose(state[0], 0.)
    assert np.all(state[1] == obs0)

    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, 0, 0., terminal1)
    state = np.array(memory.get_recent_state(obs1))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs0)
    assert np.all(state[1] == obs1)

    memory.append(obs1, 0, 0., terminal2)
    state = np.array(memory.get_recent_state(obs2))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs1)
    assert np.all(state[1] == obs2)

    memory.append(obs2, 0, 0., terminal3)
    state = np.array(memory.get_recent_state(obs3))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs2)
    assert np.all(state[1] == obs3)

    memory.append(obs3, 0, 0., terminal4)
    state = np.array(memory.get_recent_state(obs4))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs3)
    assert np.all(state[1] == obs4)

    memory.append(obs4, 0, 0., terminal5)
    state = np.array(memory.get_recent_state(obs5))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs4)
    assert np.all(state[1] == obs5)

    memory.append(obs5, 0, 0., terminal6)
    state = np.array(memory.get_recent_state(obs6))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs5)
    assert np.all(state[1] == obs6)


def test_sampling():
    memory = SequentialMemory(100, window_length=2, ignore_episode_boundaries=False)
    obs_size = (3, 4)
    actions = range(5)
    
    obs0 = np.random.random(obs_size)
    terminal0 = False
    action0 = np.random.choice(actions)
    reward0 = np.random.random()
    
    obs1 = np.random.random(obs_size)
    terminal1 = False
    action1 = np.random.choice(actions)
    reward1 = np.random.random()
    
    obs2 = np.random.random(obs_size)
    terminal2 = False
    action2 = np.random.choice(actions)
    reward2 = np.random.random()
    
    obs3 = np.random.random(obs_size)
    terminal3 = True
    action3 = np.random.choice(actions)
    reward3 = np.random.random()

    obs4 = np.random.random(obs_size)
    terminal4 = False
    action4 = np.random.choice(actions)
    reward4 = np.random.random()

    obs5 = np.random.random(obs_size)
    terminal5 = False
    action5 = np.random.choice(actions)
    reward5 = np.random.random()

    obs6 = np.random.random(obs_size)
    terminal6 = False
    action6 = np.random.choice(actions)
    reward6 = np.random.random()
    
    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, action0, reward0, terminal1)
    memory.append(obs1, action1, reward1, terminal2)
    memory.append(obs2, action2, reward2, terminal3)
    memory.append(obs3, action3, reward3, terminal4)
    memory.append(obs4, action4, reward4, terminal5)
    memory.append(obs5, action5, reward5, terminal6)
    assert memory.nb_entries == 6

    experiences = memory.sample(batch_size=5, batch_idxs=[0, 1, 2, 3, 4])
    assert len(experiences) == 5

    assert_allclose(experiences[0].state0, np.array([np.zeros(obs_size), obs0]))
    assert_allclose(experiences[0].state1, np.array([obs0, obs1]))
    assert experiences[0].action == action0
    assert experiences[0].reward == reward0
    assert experiences[0].terminal1 is False

    assert_allclose(experiences[1].state0, np.array([obs0, obs1]))
    assert_allclose(experiences[1].state1, np.array([obs1, obs2]))
    assert experiences[1].action == action1
    assert experiences[1].reward == reward1
    assert experiences[1].terminal1 is False

    assert_allclose(experiences[2].state0, np.array([obs1, obs2]))
    assert_allclose(experiences[2].state1, np.array([obs2, obs3]))
    assert experiences[2].action == action2
    assert experiences[2].reward == reward2
    assert experiences[2].terminal1 is True

    # Next experience has been re-sampled since since state0 would be terminal in which case we
    # cannot really have a meaningful transition because the environment gets reset. We thus
    # just ensure that state0 is not terminal.
    assert not np.all(experiences[3].state0 == np.array([obs2, obs3]))

    assert_allclose(experiences[4].state0, np.array([np.zeros(obs_size), obs4]))
    assert_allclose(experiences[4].state1, np.array([obs4, obs5]))
    assert experiences[4].action == action4
    assert experiences[4].reward == reward4
    assert experiences[4].terminal1 is False
    

if __name__ == '__main__':
    pytest.main([__file__])
