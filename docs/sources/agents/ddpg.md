### Introduction

---

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/ddpg.py#L22)</span>
### DDPGAgent

```python
rl.agents.ddpg.DDPGAgent(nb_actions, actor, critic, critic_action_input, memory, gamma=0.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, train_interval=1, memory_interval=1, delta_range=None, delta_clip=inf, random_process=None, custom_model_objects={}, target_model_update=0.001)
```

Write me


---

### References
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al., 2015
