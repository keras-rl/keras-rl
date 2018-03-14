<span style="float:right;">[[source]](https://github.com/keras-rl/keras-rl/blob/master/rl/processors.py#L7)</span>
### MultiInputProcessor

```python
rl.processors.MultiInputProcessor(nb_inputs)
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

<span style="float:right;">[[source]](https://github.com/keras-rl/keras-rl/blob/master/rl/processors.py#L40)</span>
### WhiteningNormalizerProcessor

```python
rl.processors.WhiteningNormalizerProcessor()
```

Normalizes the observations to have zero mean and standard deviation of one,
i.e. it applies whitening to the inputs.

This typically helps significantly with learning, especially if different dimensions are
on different scales. However, it complicates training in the sense that you will have to store
these weights alongside the policy if you intend to load it later. It is the responsibility of
the user to do so.

