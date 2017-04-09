<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L11)</span>
### Agent

```python
rl.core.Agent(processor=None)
```

Abstract base class for all implemented agents.

Each agent interacts with the environment (as defined by the `Env` class) by first observing the
state of the environment. Based on this observation the agent changes the environment by performing
an action.

Do not use this abstract base class directly but instead use one of the concrete agents implemented.
Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
interface, you can use them interchangeably.

To implement your own agent, you have to implement the following methods:

- `forward`
- `backward`
- `compile`
- `load_weights`
- `save_weights`
- `layers`

__Arguments__

- __processor__ (`Processor` instance): See [Processor](#processor) for details.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L454)</span>
### Processor

```python
rl.core.Processor()
```

Abstract base class for implementing processors.

A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
be necessary if your agent has different requirements with respect to the form of the
observations, actions, and rewards of the environment. By implementing a custom processor,
you can effectively translate between the two without having to change the underlaying
implementation of the agent or environment.

Do not use this abstract base class directly but instead use one of the concrete implementations
or write your own.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L529)</span>
### MultiInputProcessor

```python
rl.core.MultiInputProcessor(nb_inputs)
```

Converts observations from an environment with multiple observations for use in a neural network
policy.

In some cases, you have environments that return multiple different observations per timestep 
(in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
be used to report the angles for each joint). Usually, this can be handled by a policy that has
multiple inputs, one for each modality. However, observations are returned by the environment
in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
This processor converts observations appropriate for this use case.

__Arguments__

- __nb_inputs__ (integer): The number of inputs, that is different modalities, to be used.
	Your neural network that you use for the policy must have a corresponding number of
	inputs.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L566)</span>
### Env

```python
rl.core.Env()
```

The abstract environment class that is used by all agents. This class has the exact
same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
OpenAI Gym implementation, this class only defines the abstract methods without any actual
implementation.

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/rl/core.py#L642)</span>
### Space

```python
rl.core.Space()
```

Abstract model for a space that is used for the state and action spaces. This class has the
exact same API that OpenAI Gym uses so that integrating with it is trivial.

