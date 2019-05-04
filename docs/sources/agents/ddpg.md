### Introduction

---

<span style="float:right;">[[source]](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/ddpg.py#L22)</span>
### DDPGAgent

```python
rl.agents.ddpg.DDPGAgent(nb_actions, actor, critic, critic_action_input, memory, gamma=0.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, train_interval=1, memory_interval=1, delta_range=None, delta_clip=inf, random_process=None, custom_model_objects={}, target_model_update=0.001)
```

The Deep Deterministic Policy Gradient (DDPG) agent is an off policy algorithm and can be thought of as DQN for continuous action spaces. It learns a policy (the actor) and a Q-function (the critic). The policy is deterministic and its parameters are updated based on applying the chain rule to the Q-function learnt (expected reward). The Q-function is updated based on the Bellman equation, as in Q learning.

The input of the actor model should be the state observation, while its output is action itself. (Note that the action being fed as the input in the step function of the environment is therefore the output of the actor model whereas in DQN with Discrete spaces, the policy selects one action out of the nb_actions based on the model which learns the Q function.) 

The input of the critic model should be a concatenation of the state observation and the action that the actor model chooses based on this state, while its output gives the Q value for each action and state. The Keras input layer of shape nb_actions is passed as the argument critic_action_input. 

In order to balance exploitation and exploration, we can introduce a random_process which adds noise to the action determined by the actor model and allows for exploration. In the original paper, the Ornstein-Uhlenbeck process is used, which is adapted for physical control problems with inertia. 

Similar to DQN, DDPG also uses replay buffers and target networks. 

For more details, have a look at the DDPG pendulum example. 

---

### References
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al., 2015
