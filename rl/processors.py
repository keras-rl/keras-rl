import numpy as np

from rl.core import Processor
from rl.util import WhiteningNormalizer


class MultiInputProcessor(Processor):
    """Converts observations from an environment with multiple observations for use in a neural network
    policy.
    In some cases, you have environments that return multiple different observations per timestep 
    (in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
    be used to report the angles for each joint). Usually, this can be handled by a policy that has
    multiple inputs, one for each modality. However, observations are returned by the environment
    in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
    expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
    This processor converts observations appropriate for this use case.
    # Arguments
        nb_inputs (integer): The number of inputs, that is different modalities, to be used.
            Your neural network that you use for the policy must have a corresponding number of
            inputs.
    """
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        if hasattr(state_batch, 'shape'):
            if state_batch.shape >= (1,1):
                if isinstance(state_batch[0][0], dict):
                    return self.handle_dict(state_batch)
                if state_batch[0][0].ndim == 0:
                    if isinstance(state_batch[0][0].item(), dict):
                        return self.handle_dict(state_batch)
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]

    def handle_dict(self,state_batch):
        """Handles dict-like observations"""

        names = state_batch[0][0].keys()
        ordered_dict = dict()
        for key in names:
            dim = len(state_batch[0][0][key].shape)
            order_dim = state_batch.shape
            for dim_count in range(dim):
                order_dim = order_dim + (state_batch[0][0][key].shape[dim_count],)
            order = np.zeros(order_dim)

            for idx_state, state in enumerate(state_batch):
                for idx_window in range(state_batch.shape[1]):
                    for i in range(order.shape[2]):
                        if not len(state[idx_window]) == self.nb_inputs:
                            raise AssertionError()
                        order[idx_state, idx_window, i] = state_batch[idx_state][idx_window][key][i]
            ordered_dict[key] = order

        return ordered_dict


class WhiteningNormalizerProcessor(Processor):
    """Normalizes the observations to have zero mean and standard deviation of one,
    i.e. it applies whitening to the inputs.

    This typically helps significantly with learning, especially if different dimensions are
    on different scales. However, it complicates training in the sense that you will have to store
    these weights alongside the policy if you intend to load it later. It is the responsibility of
    the user to do so.
    """
    def __init__(self):
        self.normalizer = None

    def process_state_batch(self, batch):
        if self.normalizer is None:
            self.normalizer = WhiteningNormalizer(shape=batch.shape[1:], dtype=batch.dtype)
        self.normalizer.update(batch)
        return self.normalizer.normalize(batch)
