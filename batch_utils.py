"""Collection of functions to process mini batches."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_batch(training_data, batch_size, batch_counter, n_nodes):
    """
    Make a batch of morphological and geometrical data.

    Parameters
    -----------
    training_data: dict of dicts
        each inner dict is an array
        'geometry': 3-d arrays (locations)
            n_samples x n_nodes - 1 x 3
        'morphology': 2-d arrays
            n_samples x n_nodes - 2 (prufer sequences)
        example: training_data['geometry']['n20'][0:10, :, :]
                gives the geometry for the first 10 neurons
                training_data['geometry']['n20'][0:10, :]
                gives the prufer sequences for the first 10 neurons
                here, 'n20' indexes a key corresponding to
                20-node downsampled neurons.
    batch_size: int
         batch size.
     batch_counter: the index of the selected batches
         the data for batch are selected from the index
         (batch_counter - 1) * batch_size to
         batch_counter * batch_size of whole data.
     n_nodes: int
         subsampled resolution of the neurons.

    Returns
    -------
    X_locations_real: an array of size (batch_size x n_nodes - 1 x 3)
        the location of the nodes of the neuorns.
    X_prufer_real: an array of size (batch_size x n_nodes x n_nodes - 2)
        the prufer code for morphology of the neuron.
    """
    enc = OneHotEncoder(n_values=n_nodes)
    select = range((batch_counter - 1) * batch_size,
                   batch_counter * batch_size)
    tmp = np.reshape(training_data['morphology']['n'+str(n_nodes)][select, :],
                     [1, (n_nodes - 2) * batch_size])

    X_prufer_real = np.reshape(enc.fit_transform(tmp).toarray(),
                               [batch_size, n_nodes - 2, n_nodes])
    #X_prufer_real = np.swapaxes(X_prufer_real, 1, 2)

    X_locations_real = \
        training_data['geometry']['n'+str(20)][select, :, :]

    return X_locations_real, X_prufer_real


def gen_batch(geom_model,
              morph_model,
              batch_size=64,
              n_nodes=20,
              level=1,
              input_dim=100):
    """
    Generate a batch of samples from generators.

    Parameters
    ----------
    batch_size: int
        batch size
    n_nodes: list of ints
        number of nodes at each level
    level: int
        indicator of level in the hierarchy
    input_dim: int
        dimensionality of noise input
    geom_model: list of keras objects
        geometry generator for each level in the hierarchy
    morph_model: list of keras objects
        morphology generator for each level in the hierarchy

    Returns
    -------
    locations: float (batch_size x 3 x n_nodes[level] - 1)
        batch of generated locations
    prufer: float (batch_size x n_nodes[level] x n_nodes[level] - 2)
        batch of generated morphology
    """
    locations = None
    prufer = None
    for l in range(0, level + 1):

        # Generate noise code
        noise_code = np.random.randn(batch_size, 1, input_dim)

        if l == 0:
            # Generate geometry and morphology
            locations = geom_model[l].predict(noise_code)
            prufer = morph_model[l].predict(noise_code)
        else:
            # Assign previous level's geometry and morphology
            # as priors for the next level
            locations_prior = locations
            prufer_prior = prufer

            # Generate geometry and morphology conditioned on
            # the previous level
            locations = \
                geom_model[l].predict([locations_prior,
                                       prufer_prior,
                                       noise_code])
            prufer = \
                morph_model[l].predict([locations_prior,
                                        prufer_prior,
                                        noise_code])

    return locations, prufer