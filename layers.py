"""Collection of custom Keras layers."""

# Imports
from keras import backend as K
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda, Dropout
from keras.layers import Input, merge
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


# Apply batch symmetrization (A + A.T)
def batch_symmetrize(input_matrix, batch_size, n_nodes):
    """
    Take an n_nodes - 1 x n_nodes matrix and symmetrizes it.

    It concatenates a row of zeros with the matrix,
    adds the transpose and then removes the padded row.

    Parameters
    ----------
    input_matrix: theano tensor
        batch_size x n_nodes - 1 x n_nodes
    batch_size: int
        batch size
    n_nodes: int
        number of nodes of the matrix
    """
    input_matrix = K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]),
                                  input_matrix], axis=1)
    result, updates = \
        K.theano.scan(fn=lambda n: input_matrix[n, :, :] +
                      input_matrix[n, :, :].T,
                      sequences=K.arange(input_matrix.shape[0]))
    return result[:, 1:, :]


# Masked softmax Lambda layer
def masked_softmax(input_layer, n_nodes, batch_size):
    """
    A Lambda layer to mask a matrix of outputs to be lower-triangular.

    Each row must sum up to one. We apply a lower triangular mask of ones
    and then add an upper triangular mask of a large negative number.

    Parameters
    ----------
    input_layer: keras layer object
        (n x 1, n) matrix
    n_nodes: int
        number of nodes
    batch_size: int
        batch size

    Returns
    -------
    output_layer: keras layer object
        (n x 1, n) matrix
    """
    input_layer = batch_symmetrize(input_layer, batch_size, n_nodes)
    mask_lower = K.theano.tensor.tril(K.ones((n_nodes - 1, n_nodes)))
    mask_upper = \
        K.theano.tensor.triu(-100. * K.ones((n_nodes - 1, n_nodes)), 1)
    mask_layer = mask_lower * input_layer + mask_upper
    mask_layer = mask_layer + 2 * K.eye(n_nodes)[0:n_nodes - 1, 0:n_nodes]
    mask_layer = \
        K.reshape(mask_layer, (batch_size * (n_nodes - 1), n_nodes))
    softmax_layer = K.softmax(mask_layer)
    output_layer = K.reshape(softmax_layer, (batch_size, n_nodes - 1, n_nodes))
    return output_layer


# Compute full adjacency matrix
def full_matrix(adjacency, n_nodes):
    """
    Returning the full adjacency matrix of adjacency.

    Parameters
    ----------
    adjacency: keras layer object
        (n , n) matrix
    Returns
    -------
    keras layer object
        (n , n) matrix
    """
    return K.theano.tensor.nlinalg.matrix_inverse(K.eye(n_nodes) - adjacency)


# Masked softmax Lambda layer
def masked_softmax_full(input_layer, n_nodes, batch_size):
    """
    A Lambda layer to compute a lower-triangular version of the full adjacency.

    Each row must sum up to one. We apply a lower triangular mask of ones
    and then add an upper triangular mask of a large negative number.
    After that we return the full adjacency matrix.

    Parameters
    ----------
    input_layer: keras layer object
        (n x 1, n) matrix

    Returns
    -------
    output_layer: keras layer object
        (n x 1, n) matrix
    """
    mask_layer = masked_softmax(input_layer, n_nodes, batch_size)
    mask_layer = \
        K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]), mask_layer],
                      axis=1)
    result, updates = \
        K.theano.scan(fn=lambda n: full_matrix(mask_layer[n, :, :], n_nodes),
                      sequences=K.arange(batch_size))
    return result[:, 1:, :]


def distance_from_parent(adjacency, locations, n_nodes, batch_size):
    """
    Return distance from parent.

    Parameters
    ----------
    adjacency: theano/keras tensor
        (batch_size x n_nodes - 1 x n_nodes) matrix
    locations: theano/keras tensor
        (batch_size x n_nodes x 3) matrix

    Returns
    -------
    result: keras layer object
        (batch_size x n_nodes - 1 x n_nodes) matrix
    """
    result, updates = \
        K.theano.scan(fn=lambda n: K.dot(K.eye(n_nodes) - adjacency[n, :, :],
                                         locations[n, :, :]),
                      sequences=K.arange(batch_size))
    return result


# Embedding layers
def embedder(geometry_input,
             morphology_input,
             n_nodes=10,
             hidden_dim=20,
             embedding_dim=100):
    """
    Joint embedding of geometric coordinates and tree morphology.

    Parameters
    ----------
    geometry_input: keras layer object
        geometry Input layer
    morphology_input: keras layer object
        morphology Input layer object
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimensions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    embedding: keras layer object
        embedding layer
    """
    # Merge
    merged_layer = merge([geometry_input,
                          morphology_input], mode='concat')

    LSTM
    embedding_lstm1 = \
        LSTM(input_dim=(n_nodes + 3),
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             W_regularizer=l2(0.1),
             U_regularizer=l2(0.1),
             return_sequences=True)(merged_layer)
    # embedding_lstm1 = BatchNormalization()(embedding_lstm1)

    embedding_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(embedding_lstm1)

    # embedding_reshaped = \
    #     Reshape(target_shape=
    #             (1, (n_nodes - 1) * (n_nodes + 3)))(merged_layer)

    embedding = Dense(input_dim=(n_nodes - 1) * hidden_dim,
                      output_dim=embedding_dim,
                      W_regularizer=l2(0.01),
                      name='embedding')(embedding_reshaped)
    # embedding = BatchNormalization()(embedding)
    return embedding


def geometry_embedder(geometry_input,
                      n_nodes=10,
                      hidden_dim=20,
                      embedding_dim=100):
    """
    Embedding of geometric coordinates of nodes.

    Parameters
    ----------
    geometry_input: keras layer object
        input layer
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimensions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    geometry_embedding: keras layer object
        embedding layer
    """
    # LSTM
    geometry_embedding_lstm1 = \
        LSTM(input_dim=3,
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             W_regularizer=l2(0.1),
             U_regularizer=l2(0.1),
             return_sequences=True)(geometry_input)
    # geometry_embedding_lstm1 = BatchNormalization()(geometry_embedding_lstm1)

    geometry_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(geometry_embedding_lstm1)
    geometry_embedding = Dense(input_dim=(n_nodes - 1) * hidden_dim,
                               output_dim=embedding_dim,
                               W_regularizer=l2(0.01),
                               name='geometry_embedding')(geometry_reshaped)
    # geometry_embedding = BatchNormalization()(geometry_embedding)

    return geometry_embedding


def morphology_embedder(morphology_input,
                        n_nodes=10,
                        hidden_dim=20,
                        embedding_dim=100):
    """
    Embedding of tree morphology (softmax parent code).

    Parameters
    ----------
    morphology_input: keras layer object
        input layer
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimeisions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    morphology_embedding: keras layer object
        embedding layer
    """
    # LSTM
    morphology_embedding_lstm1 = \
        LSTM(input_dim=n_nodes,
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             W_regularizer=l2(0.1),
             U_regularizer=l2(0.1),
             return_sequences=True)(morphology_input)
    # morphology_embedding_lstm1 = \
    #     BatchNormalization()(morphology_embedding_lstm1)

    morphology_embedding_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(morphology_embedding_lstm1)

    morphology_embedding = \
        Dense(input_dim=(n_nodes - 1) * n_nodes,
              output_dim=embedding_dim,
              W_regularizer=l2(0.01),
              name='morphology_embedding')(morphology_embedding_reshaped)
    # morphology_embedding = BatchNormalization()(morphology_embedding)

    return morphology_embedding


def feature_extractor(inputs,
                      n_nodes,
                      batch_size):
    """
    Compute various features and concatenate them.

    Parameters
    ----------
    morphology_input: keras layer object
        (batch_size x n_nodes - 1 x n_nodes)
        the adjacency matrix of each sample.

    geometry_input: keras layer object
        (batch_size x n_nodes - 1 x 3)
        the locations of each nodes.

    n_nodes: int
        number of nodes

    batch_size: int
        batch size

    Returns
    -------
    features: keras layer object
        (batch_size x n_nodes x n_features)
        The features currently supports:
            - The adjacency
            - The full adjacency
            - locations
            - distance from imediate parents
    """
    geometry_input, morphology_input = inputs
    adjacency = \
        masked_softmax(morphology_input, n_nodes, batch_size)
    adjacency = \
        K.concatenate([K.zeros(shape=(batch_size, 1, n_nodes)),
                       adjacency], axis=1)
    full_adjacency = \
        masked_softmax_full(morphology_input, n_nodes, batch_size)
    full_adjacency = \
        K.concatenate([K.zeros(shape=(batch_size, 1, n_nodes)),
                       full_adjacency], axis=1)
    geometry_input = K.concatenate([K.zeros(shape=(batch_size, 1, 3)),
                                    geometry_input], axis=1)
    # distance = distance_from_parent(adjacency,
    #                                 geometry_input,
    #                                 n_nodes,
    #                                 batch_size)
    features = K.concatenate([adjacency,
                              full_adjacency,
                              geometry_input], axis=2)
    return features
