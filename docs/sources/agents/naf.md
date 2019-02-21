### Introduction

---

<span style="float:right;">[[source]](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py#L548)</span>
### NAFAgent

```python
rl.agents.dqn.NAFAgent(V_model, L_model, mu_model, random_process=None, covariance_mode='full')
```

Normalized Advantage Function (NAF) agents is a way of extending DQN to a continuous action space, and is simpler than DDPG agents. 

The Q-function is here decomposed into an advantage term A and state value term V. The agent thus makes use of three models: the V_model learns the state value term, while the advantage term A is constructed based on the L_model and the mu_model such that the mu_model is always the action that maximizes the Q function. (exact mathematical formulation in the paper)

Since the mu_model chooses the action deterministically, we can add a random_process to balance exploration and exploitation. Similar to DQN, we use target networks for stability and keep a replay buffer. 


---

### References
- [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748), Gu et al., 2016
