## Available Agents

| Name                   | Implementation         | Observation Space  | Action Space   | 
| ---------------------- |------------------------| -------------------| ---------------|
| [DQN](/agents/dqn)     | `rl.agents.DQNAgent`   | discrete or continuous | discrete   | 
| [DDPG](/agents/ddpg)   | `rl.agents.DDPGAgent`  | discrete or continuous | continuous | 
| [NAF](/agents/naf)     | `rl.agents.NAFAgent`   | discrete or continuous | continuous |
| [CEM](/agents/cem)     | `rl.agents.CEMAgent`   | discrete or continuous | discrete   |
| [SARSA](/agents/sarsa) | `rl.agents.SARSAAgent` | discrete or continuous | discrete   | 

---

## Common API

All agents share a common API. This allows you to easily switch between different agents.
That being said, keep in mind that some agents make assumptions regarding the action space, i.e. assume discrete
or continuous actions.

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L44)</span>

### fit


```python
fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1, visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000, nb_max_episode_steps=None)
```


Trains the agent on the given environment.

__Arguments__

- __env:__ (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
- __nb_steps__ (integer): Number of training steps to be performed.
- __action_repetition__ (integer): Number of times the agent repeats the same action without
	observing the environment again. Setting this to a value > 1 can be useful
	if a single action only has a very small effect on the environment.
- __callbacks__ (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
	List of callbacks to apply during training. See [callbacks](/callbacks) for details.
- __verbose__ (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
- __visualize__ (boolean): If `True`, the environment is visualized during training. However,
	this is likely going to slow down training significantly and is thus intended to be
	a debugging instrument.
- __nb_max_start_steps__ (integer): Number of maximum steps that the agent performs at the beginning
	of each episode using `start_step_policy`. Notice that this is an upper limit since
	the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
	at the beginning of each episode.
- __start_step_policy__ (`lambda observation: action`): The policy
	to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
- __log_interval__ (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
- __nb_max_episode_steps__ (integer): Number of steps per episode that the agent performs before
	automatically resetting the environment. Set to `None` if each episode should run
	(potentially indefinitely) until the environment signals a terminal state.

__Returns__

A `keras.callbacks.History` instance that recorded the entire training process.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L231)</span>

### test


```python
test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True, nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1)
```


Callback that is called before training begins."

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L391)</span>

### compile


```python
compile(self, optimizer, metrics=[])
```


Compiles an agent and the underlaying models to be used for training and testing.

__Arguments__

- __optimizer__ (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
- __metrics__ (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L39)</span>

### get_config


```python
get_config(self)
```


Configuration of the agent for serialization.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L364)</span>

### reset_states


```python
reset_states(self)
```


Resets all internally kept states after an episode is completed.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L400)</span>

### load_weights


```python
load_weights(self, filepath)
```


Loads the weights of an agent from an HDF5 file.

__Arguments__

- __filepath__ (str): The path to the HDF5 file.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L408)</span>

### save_weights


```python
save_weights(self, filepath, overwrite=False)
```


Saves the weights of an agent as an HDF5 file.

__Arguments__

- __filepath__ (str): The path to where the weights should be saved.
- __overwrite__ (boolean): If `False` and `filepath` already exists, raises an error.

