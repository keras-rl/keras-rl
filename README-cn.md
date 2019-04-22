# Keras实现深度强化学习
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


## keras-rl是什么？
[English](README.md)

`keras-rl` 使用python实现了一些经典的深度强化学习算法并且能与深度学习库[Keras](http://keras.io)无缝集成.

此外, `keras-rl` 和[OpenAI Gym](https://gym.openai.com/)一起开箱即用. 这意味着你想要评估或者训练不同的算法变得非常简单.

当然你也可以根据你自己的需要扩展 `keras-rl`. 你可以使用内置的 Keras 回调函数和指标或者完全自己定义.
更重要的是, 通过简单的扩展一些抽象类，就很容易实现自己的环境甚至是算法. [点击查看在线文档](http://keras-rl.readthedocs.org).


## keras-rl包含了哪些?
直到今天截止，以下算法已经被实现：

- [x] Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](https://www.nature.com/articles/nature14236)
- [x] Double DQN [[3]](http://arxiv.org/abs/1509.06461)
- [x] Deep Deterministic Policy Gradient (DDPG) [[4]](http://arxiv.org/abs/1509.02971)
- [x] Continuous DQN (CDQN or NAF) [[6]](http://arxiv.org/abs/1603.00748)
- [x] Cross-Entropy Method (CEM) [[7]](http://learning.mpi-sws.org/mlss2016/slides/2016-MLSS-RL.pdf), [[8]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf)
- [x] Dueling network DQN (Dueling DQN) [[9]](https://arxiv.org/abs/1511.06581)
- [x] Deep SARSA [[10]](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)
- [ ] Asynchronous Advantage Actor-Critic (A3C) [[5]](http://arxiv.org/abs/1602.01783)
- [ ] Proximal Policy Optimization Algorithms (PPO) [[11]](https://arxiv.org/abs/1707.06347)

更多关于agent的信息请查看 [文档](http://keras-rl.readthedocs.io/en/latest/agents/overview/).


## 安装

- 从Pypi安装 (推荐):

```
pip install keras-rl
```

- 从github源码安装:

```
git clone https://github.com/keras-rl/keras-rl.git
cd keras-rl
python setup.py install
```

## 示例

如果你想要运行代码示例，你还需要安装:
- **gym** by OpenAI: [安装说明](https://github.com/openai/gym#installation)
- **h5py**: 直接运行 `pip install h5py`

如果是atari示例, 你还需要安装:
- **Pillow**: `pip install Pillow`
- **gym[atari]**: gym Atari 模块. Use `pip install gym[atari]`

安装完所有内容后，您可以尝试一个简单的示例
```bash
python examples/dqn_cartpole.py
```
这是一个非常简单的例子，它应该会相对快速地完成收敛，恭喜你已经入门了！
它还可以在训练期间进行可视化，所以您可以观看它是如何学习的。 很酷吧？

更多的已经训练好的[模型参数](https://github.com/matthiasplappert/keras-rl-weights).

If you have questions or problems, please file an issue or, even better, fix the problem yourself and submit a pull request!
如果您有任何疑问或问题，请在github上提出issue，或者更好的是解决它并提交PR！

## 扩展项目

- [星际争霸2环境](https://soygema.github.io/starcraftII_machine_learning/#0)

你是否正在项目中使用Keras-RL?告诉我们！

## 引用

如果你在研究中使用了`keras-rl`, 你需要声明以下引用:
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

## 参考文献

1. *Playing Atari with Deep Reinforcement Learning*, Mnih et al., 2013
2. *Human-level control through deep reinforcement learning*, Mnih et al., 2015
3. *Deep Reinforcement Learning with Double Q-learning*, van Hasselt et al., 2015
4. *Continuous control with deep reinforcement learning*, Lillicrap et al., 2015
5. *Asynchronous Methods for Deep Reinforcement Learning*, Mnih et al., 2016
6. *Continuous Deep Q-Learning with Model-based Acceleration*, Gu et al., 2016
7. *Learning Tetris Using the Noisy Cross-Entropy Method*, Szita et al., 2006
8. *Deep Reinforcement Learning (MLSS lecture notes)*, Schulman, 2016
9. *Dueling Network Architectures for Deep Reinforcement Learning*, Wang et al., 2016
10. *Reinforcement learning: An introduction*, Sutton and Barto, 2011
11. *Proximal Policy Optimization Algorithms*, Schulman et al., 2017
