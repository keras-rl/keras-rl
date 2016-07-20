# Deep Reinforcement Learning for Keras

![cartpole](/assets/cartpole.gif?raw=true) ![pendulum](/assets/pendulum.gif?raw=true)

## What is it?
`keras-rl` implements some state-of-the art deep reinforcement learning algorithms in Python and seamlessly integrates with the deep learning library [Keras](http://keras.io). Just like Keras, it works with either [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/), which means that you can train your algorithm efficiently either on CPU or GPU.
Furthermore, `keras-rl` works with [OpenAI Gym]() out of the box. This means that evaluating and playing around with different algorithms is easy.
Of course you can extend `keras-rl` according to your own needs. You can use built-in Keras callbacks and metrics or define your own.
Even more so, it is easy to implement your own environments and even algorithms by simply extending some simple abstract classes.

In a nutshell: `keras-rl` makes it really easy to run state-of-the-art deep reinforcement learning algorithms, uses Keras and thus Theano and TensorFlow and was built with OpenAI Gym in mind.

## What is included?
As of today, the following algorithms have been implemented:
- Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)
- Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- Deep Deterministic Policy Gradient (DDPG) [[4]](http://arxiv.org/pdf/1509.02971v2.pdf)

## Requirements
- Python 2.7
- [Keras](http://keras.io) >= 1.0

That's it. However, if you want to run the examples, you'll also need the following dependencies:
- [OpenAI Gym](https://github.com/openai/gym)
- [h5py](https://pypi.python.org/pypi/h5py)

`keras-rl` also works with [TensorFlow](https://www.tensorflow.org/). To find out how to use TensorFlow instead of [Theano](http://deeplearning.net/software/theano/), please refer to the [Keras documentation](http://keras.io/#switching-from-theano-to-tensorflow).

## Acknowledgments
The foundation for this library was developed during my work at the [High Performance Humanoid Technologies (H²T)](https://h2t.anthropomatik.kit.edu/) lab at the [Karlsruhe Institute of Technologie (KIT)](https://kit.edu).
It has since been adapted to become a general-purpose library.

## Todos
- Tests: I haven't yet had time to get started, but this is important.
- Documentation: Currently, the documentation is pretty much non-existent.
- TROP, priority-based memory, dueling DQN, A3C, async DQN, ...
