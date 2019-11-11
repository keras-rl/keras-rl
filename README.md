# Deep Reinforcement Learning for Keras

[![Build Status](https://api.travis-ci.org/keras-rl/keras-rl.svg?branch=master)](https://travis-ci.org/keras-rl/keras-rl)
[![Documentation](https://readthedocs.org/projects/keras-rl/badge/)](http://keras-rl.readthedocs.io/)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-rl/keras-rl/blob/master/LICENSE)
[![Join the chat at https://gitter.im/keras-rl/Lobby](https://badges.gitter.im/keras-rl/Lobby.svg)](https://gitter.im/keras-rl/Lobby)

<table>
  <tr>
    <td><img src="/assets/breakout.gif?raw=true" width="200"></td>
    <td><img src="/assets/cartpole.gif?raw=true" width="200"></td>
    <td><img src="/assets/pendulum.gif?raw=true" width="200"></td>
  </tr>
</table>

## What is it?

`keras-rl` implements some state-of-the art deep reinforcement learning algorithms in Python and seamlessly integrates with the deep learning library [Keras](http://keras.io).

Furthermore, `keras-rl` works with [OpenAI Gym](https://gym.openai.com/) out of the box. This means that evaluating and playing around with different algorithms is easy.

Of course you can extend `keras-rl` according to your own needs. You can use built-in Keras callbacks and metrics or define your own.
Even more so, it is easy to implement your own environments and even algorithms by simply extending some simple abstract classes. Documentation is available [online](http://keras-rl.readthedocs.org).

## What is included?

As of today, the following algorithms have been implemented:

- [x] Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](https://www.nature.com/articles/nature14236)
- [x] Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- [x] Deep Deterministic Policy Gradient (DDPG) [[4]](http://arxiv.org/abs/1509.02971)
- [x] Continuous DQN (CDQN or NAF) [[6]](http://arxiv.org/abs/1603.00748)
- [x] Cross-Entropy Method (CEM) [[7]](http://learning.mpi-sws.org/mlss2016/slides/2016-MLSS-RL.pdf), [[8]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf)
- [x] Dueling network DQN (Dueling DQN) [[9]](https://arxiv.org/abs/1511.06581)
- [x] Deep SARSA [[10]](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)
- [ ] Asynchronous Advantage Actor-Critic (A3C) [[5]](http://arxiv.org/abs/1602.01783)
- [ ] Proximal Policy Optimization Algorithms (PPO) [[11]](https://arxiv.org/abs/1707.06347)

You can find more information on each agent in the [doc](http://keras-rl.readthedocs.io/en/latest/agents/overview/).

## Installation

- Install Keras-RL from Pypi (recommended):

```
pip install keras-rl
```

- Install from Github source:

```
git clone https://github.com/keras-rl/keras-rl.git
cd keras-rl
python setup.py install
```

## Examples

If you want to run the examples, you'll also have to install:

- **gym** by OpenAI: [Installation instruction](https://github.com/openai/gym#installation)
- **h5py**: simply run `pip install h5py`

For atari example you will also need:

- **Pillow**: `pip install Pillow`
- **gym[atari]**: Atari module for gym. Use `pip install gym[atari]`

Once you have installed everything, you can try out a simple example:

```bash
python examples/dqn_cartpole.py
```

This is a very simple example and it should converge relatively quickly, so it's a great way to get started!
It also visualizes the game during training, so you can watch it learn. How cool is that?

Some sample weights are available on [keras-rl-weights](https://github.com/matthiasplappert/keras-rl-weights).

If you have questions or problems, please file an issue or, even better, fix the problem yourself and submit a pull request!

## External Projects

- [Starcraft II Learning Environment](https://soygema.github.io/starcraftII_machine_learning/#0)

You're using Keras-RL on a project? Open a PR and share it!

## Visualizing Training Metrics

To see graphs of your training progress and compare across runs, run `pip install wandb` and add the WandbLogger callback to your agent's `fit()` call:

```python
from rl.callbacks import WandbLogger

...

agent.fit(env, nb_steps=50000, callbacks=[WandbLogger()])
```

For more info and options, see the [W&B docs](https://docs.wandb.com/getting-started).

## Citing

If you use `keras-rl` in your research, you can cite it as follows:

```bibtex
@misc{plappert2016kerasrl,
    author = {Matthias Plappert},
    title = {keras-rl},
    year = {2016},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/keras-rl/keras-rl}},
}
```

## References

1. _Playing Atari with Deep Reinforcement Learning_, Mnih et al., 2013
2. _Human-level control through deep reinforcement learning_, Mnih et al., 2015
3. _Deep Reinforcement Learning with Double Q-learning_, van Hasselt et al., 2015
4. _Continuous control with deep reinforcement learning_, Lillicrap et al., 2015
5. _Asynchronous Methods for Deep Reinforcement Learning_, Mnih et al., 2016
6. _Continuous Deep Q-Learning with Model-based Acceleration_, Gu et al., 2016
7. _Learning Tetris Using the Noisy Cross-Entropy Method_, Szita et al., 2006
8. _Deep Reinforcement Learning (MLSS lecture notes)_, Schulman, 2016
9. _Dueling Network Architectures for Deep Reinforcement Learning_, Wang et al., 2016
10. _Reinforcement learning: An introduction_, Sutton and Barto, 2011
11. _Proximal Policy Optimization Algorithms_, Schulman et al., 2017
