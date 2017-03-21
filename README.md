# Deep Reinforcement Learning for Keras
[![Build Status](https://api.travis-ci.org/matthiasplappert/keras-rl.svg?branch=master)](https://travis-ci.org/matthiasplappert/keras-rl)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/matthiasplappert/keras-rl/blob/master/LICENSE)
[![Join the chat at https://gitter.im/keras-rl/Lobby](https://badges.gitter.im/keras-rl/Lobby.svg)](https://gitter.im/keras-rl/Lobby)


<table>
  <tr>
    <td><img src="/assets/breakout.gif?raw=true" width="200"></td>
    <td><img src="/assets/cartpole.gif?raw=true" width="200"></td>
    <td><img src="/assets/pendulum.gif?raw=true" width="200"></td>
  </tr>
</table>

## What is it?
`keras-rl` implements some state-of-the art deep reinforcement learning algorithms in Python and seamlessly integrates with the deep learning library [Keras](http://keras.io). Just like Keras, it works with either [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/), which means that you can train your algorithm efficiently either on CPU or GPU.
Furthermore, `keras-rl` works with [OpenAI Gym](https://gym.openai.com/) out of the box. This means that evaluating and playing around with different algorithms is easy.
Of course you can extend `keras-rl` according to your own needs. You can use built-in Keras callbacks and metrics or define your own.
Even more so, it is easy to implement your own environments and even algorithms by simply extending some simple abstract classes.

In a nutshell: `keras-rl` makes it really easy to run state-of-the-art deep reinforcement learning algorithms, uses Keras and thus Theano or TensorFlow and was built with OpenAI Gym in mind.

## What is included?
As of today, the following algorithms have been implemented:

- Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)
- Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- Deep Deterministic Policy Gradient (DDPG) [[4]](http://arxiv.org/abs/1509.02971)
- Continuous DQN (CDQN or NAF) [[6]](http://arxiv.org/abs/1603.00748)
- Cross-Entropy Method (CEM) [[7]](http://learning.mpi-sws.org/mlss2016/slides/2016-MLSS-RL.pdf), [[8]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf)
- Dueling network DQN (Dueling DQN) [[9]](https://arxiv.org/abs/1511.06581)
- Deep SARSA [[10]](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)

You can find more information on each agent in the [wiki](https://github.com/matthiasplappert/keras-rl/wiki/Agent-Overview).

I'm currently working on the following algorithms, which can be found on the `experimental` branch:

- Asynchronous Advantage Actor-Critic (A3C) [[5]](http://arxiv.org/abs/1602.01783)

Notice that these are **only experimental** and might currently not even run.

## How do I install it and how do I get started?
Installing `keras-rl` is easy. Just run the following commands and you should be good to go:
```bash
pip install keras-rl
```
This will install `keras-rl` and all necessary dependencies.

If you want to run the examples, you'll also have to install `gym` by OpenAI.
Please refer to [their installation instructions](https://github.com/openai/gym#installation).
It's quite easy and works nicely on Ubuntu and Mac OS X.
You'll also need the `h5py` package to load and save model weights, which can be installed using
the following command:
```bash
pip install h5py
```

Once you have installed everything, you can try out a simple example:
```bash
python examples/dqn_cartpole.py
```
This is a very simple example and it should converge relatively quickly, so it's a great way to get started!
It also visualizes the game during training, so you can watch it learn. How cool is that?

Unfortunately, he documentation of `keras-rl` is currently almost non-existent.
However, you can find a couple of more examples that illustrate the usage of both DQN (for tasks with discrete actions) as well as for DDPG (for tasks with continuous actions).
While these examples are not replacement for a proper documentation, they should be enough to get started quickly and to see the magic of reinforcement learning yourself.
I also encourage you to play around with other environments (OpenAI Gym has plenty) and maybe even try to find better hyperparameters for the existing ones.

If you have questions or problems, please file an issue or, even better, fix the problem yourself and submit a pull request!

## Do I have to train the models myself?
Training times can be very long depending on the complexity of the environment.
[This repo](https://github.com/matthiasplappert/keras-rl-weights) provides some weights that were obtained by running (at least some) of the examples that are included in `keras-rl`.
You can load the weights using the `load_weights` method on the respective agents.

## Requirements
- Python 2.7
- [Keras](http://keras.io) >= 1.0.7

That's it. However, if you want to run the examples, you'll also need the following dependencies:
- [OpenAI Gym](https://github.com/openai/gym)
- [h5py](https://pypi.python.org/pypi/h5py)

`keras-rl` also works with [TensorFlow](https://www.tensorflow.org/). To find out how to use TensorFlow instead of [Theano](http://deeplearning.net/software/theano/), please refer to the [Keras documentation](http://keras.io/#switching-from-theano-to-tensorflow).

## Support
You can ask questions and join the development discussion:

- On the [Keras-RL Google group](https://groups.google.com/forum/#!forum/keras-rl-users).
- On the [Keras-RL Gitter channel](https://gitter.im/keras-rl/Lobby).

You can also post **bug reports and feature requests** (only!) in [Github issues](https://github.com/matthiasplappert/keras-rl/issues).

## Running the Tests
To run the tests locally, you'll first have to install the following dependencies:
```bash
pip install pytest pytest-xdist pep8 pytest-pep8 pytest-cov python-coveralls
```
You can then run all tests using this command:
```bash
py.test tests/.
```
If you want to check if the files conform to the PEP8 style guidelines, run the following command:
```bash
py.test --pep8
```

## Citing
If you use `keras-rl` in your research, you can cite it as follows:
```bibtex
@misc{plappert2016kerasrl,
    author = {Matthias Plappert},
    title = {keras-rl},
    year = {2016},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/matthiasplappert/keras-rl}},
}
```


## Acknowledgments
The foundation for this library was developed during my work at the [High Performance Humanoid Technologies (H²T)](https://h2t.anthropomatik.kit.edu/) lab at the [Karlsruhe Institute of Technology (KIT)](https://kit.edu).
It has since been adapted to become a general-purpose library.

## References
1. *Playing Atari with Deep Reinforcement Learning*, Mnih et al., 2013
2. *Human-level control through deep reinforcement learning*, Mnih et al., 2015
3. *Deep Reinforcement Learning with Double Q-learning*, van Hasselt et al., 2015
4. *Continuous control with deep reinforcement learning*, Lillicrap et al., 2015
5. *Asynchronous Methods for Deep Reinforcement Learning*, Mnih et al., 2016
6. *Continuous Deep Q-Learning with Model-based Acceleration*, Gu et al., 2016
7. *Learning Tetris Using the Noisy Cross-Entropy Method*, Szita et al., 2006
8. *Deep Reinforcement Learning (MLSS lecture notes)*, John Schulman, 2016.
9. *Dueling Network Architectures for Deep Reinforcement Learning*, Ziyu Wang et al., 2016.
10. *Reinforcement learning: An introduction*, Sutton and Barto, 2011.

## Todos
- Tests: I haven't yet had time to get started, but this is important.
- Documentation: Currently, the documentation is pretty much non-existent.
- TRPO, priority-based memory, A3C, async DQN, ...
