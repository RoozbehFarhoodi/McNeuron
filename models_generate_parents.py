"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras import backend as K


# Embedding layers
def embedder(n_nodes=10, hidden_dim=20, embedding_dim=100):
    """
    Joint embedding of geometric coordinates and tree morphology.

    Parameters
    ----------
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimensions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    geometry_input: keras layer object
        input layer
    morphology_input: keras layer object
        input layer
    embedding: keras layer object
        embedding layer
    """
    geometry_input = Input(shape=(n_nodes - 1, 3))
    morphology_input = Input(shape=(n_nodes - 1, n_nodes))

    # Merge
    merged_layer = merge([geometry_input,
                          morphology_input], mode='concat')

    # LSTM
    embedding_lstm1 = \
        LSTM(input_dim=(n_nodes + 3),
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             return_sequences=True)(merged_layer)

    embedding_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(embedding_lstm1)

    embedding = Dense(input_dim=(n_nodes - 1) * hidden_dim,
                      output_dim=embedding_dim,
                      name='embedding')(embedding_reshaped)
    return geometry_input, morphology_input, embedding


def geometry_embedder(n_nodes=10, hidden_dim=20, embedding_dim=100):
    """
    Embedding of geometric coordinates of nodes.

    Parameters
    ----------
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimensions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    geometry_input: keras layer object
        input layer
    geometry_embedding: keras layer object
        embedding layer
    """
    geometry_input = Input(shape=(n_nodes - 1, 3))

    # LSTM
    geometry_embedding_lstm1 = \
        LSTM(input_dim=3,
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             return_sequences=True)(geometry_input)

    geometry_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(geometry_embedding_lstm1)
    geometry_embedding = Dense(input_dim=(n_nodes - 1) * hidden_dim,
                               output_dim=embedding_dim,
                               name='geometry_embedding')(geometry_reshaped)
    return geometry_input, geometry_embedding


def morphology_embedder(n_nodes=10, hidden_dim=20, embedding_dim=100):
    """
    Embedding of tree morphology (softmax Prufer code).

    Parameters
    ----------
    n_nodes: int
        number of nodes
    hidden_dim: int
        number of hidden dimeisions for LSTM
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    morphology_input: keras layer object
        input layer
    morphology_embedding: keras layer object
        embedding layer
    """
    morphology_input = Input(shape=(n_nodes - 1, n_nodes))

    # LSTM
    morphology_embedding_lstm1 = \
        LSTM(input_dim=n_nodes,
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_input)

    morphology_embedding_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(morphology_embedding_lstm1)

    morphology_embedding = \
        Dense(input_dim=(n_nodes - 1) * n_nodes,
              output_dim=embedding_dim,
              name='morphology_embedding')(morphology_embedding_reshaped)
    return morphology_input, morphology_embedding


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

    Returns
    -------
    output_layer: keras layer object
        (n x 1, n) matrix
    """
    mask_lower = K.theano.tensor.tril(K.ones((n_nodes - 1, n_nodes)))
    mask_upper = \
        K.theano.tensor.triu(-100. * K.ones((n_nodes - 1, n_nodes)), 1)
    mask_layer = mask_lower * K.log(input_layer) + mask_upper
    mask_layer = \
        K.reshape(mask_layer, (batch_size * (n_nodes - 1), n_nodes))
    softmax_layer = K.softmax(mask_layer)
    output_layer = K.reshape(softmax_layer, (batch_size, n_nodes - 1, n_nodes))
    return output_layer
    # return input_layer


def full_matrix(adjacency, n_nodes):
    """
    Returning the full matrix of adjacency.

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
    A Lambda layer to mask a matrix of outputs to be lower-triangular.

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
    mask_lower = K.theano.tensor.tril(K.ones((n_nodes - 1, n_nodes)))

    mask_upper = \
        K.theano.tensor.triu(-100. * K.ones((n_nodes - 1, n_nodes)), 1)

    mask_layer = mask_lower * K.log(input_layer) + mask_upper

    mask_layer = \
        K.reshape(mask_layer, (batch_size * (n_nodes - 1), n_nodes))

    softmax_layer = K.softmax(mask_layer)

    mask_layer = \
        K.reshape(softmax_layer, (batch_size, n_nodes - 1, n_nodes))

    mask_layer = \
        K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]), mask_layer],
                      axis=1)

    result, updates = \
        K.theano.scan(fn = lambda n: full_matrix(mask_layer[n, : , :], n_nodes),
                      sequences=K.arange(batch_size))

    return result[:, 1:, :]
    # return input_layer


# Generators
def generator(n_nodes_in=10,
              n_nodes_out=20,
              noise_dim=100,
              embedding_dim=100,
              hidden_dim=20,
              batch_size=64,
              use_context=True):
    """
    Generator network.

    Parameters
    ----------
    n_nodes_in: int
        number of nodes in the tree providing context input
    n_nodes_out: int
        number of nodes in the output tree
    noise_dim: int
        dimensionality of noise input
    embedding_dim: int
        dimensionality of embedding for context input
    use_context: bool
        if True, use context, else only noise input

    Returns
    -------
    geometry_model: keras model object
        model of geometry generator
    conditional_geometry_model: keras model object
        model of geometry generator conditioned on morphology
    morphology_model: keras model object
        model of morphology generator
    conditional_morphology_model: keras model object
        model of morphology generator conditioned on geometry
    """
    # Embed contextual information from previous levels
    if use_context is True:
        prior_geometry_input, \
            prior_morphology_input, \
            prior_embedding = \
            embedder(n_nodes=n_nodes_in,
                     hidden_dim=hidden_dim,
                     embedding_dim=embedding_dim)

    # Generate noise input
    noise_input = Input(shape=(1, noise_dim), name='noise_input')

    # Embed conditional information from current level
    geometry_input, geometry_embedding = \
        geometry_embedder(n_nodes=n_nodes_out,
                          hidden_dim=hidden_dim,
                          embedding_dim=embedding_dim)

    morphology_input, morphology_embedding = \
        morphology_embedder(n_nodes=n_nodes_out,
                            hidden_dim=hidden_dim,
                            embedding_dim=embedding_dim)

    # Concatenate prior context and noise inputs
    if use_context is True:
        all_common_inputs = merge([prior_embedding,
                                   noise_input], mode='concat')
    else:
        all_common_inputs = noise_input

    # ---------------
    # Geometry model
    # ---------------

    # Dense
    geometry_hidden_dim = (n_nodes_out - 1) * 3
    geometry_hidden1 = Dense(geometry_hidden_dim)(all_common_inputs)
    geometry_hidden2 = Dense(geometry_hidden_dim)(geometry_hidden1)

    # Reshape
    geometry_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, 3))(geometry_hidden2)

    # LSTM
    geometry_lstm1 = \
        LSTM(input_dim=3,
             input_length=n_nodes_out - 1,
             output_dim=3,
             return_sequences=True)(geometry_reshaped)
    geometry_lstm2 = \
        LSTM(input_dim=3,
             input_length=n_nodes_out - 1,
             output_dim=3,
             return_sequences=True)(geometry_lstm1)
    # TimeDistributed
    geometry_output = \
        TimeDistributed(Dense(input_dim=3,
                              output_dim=3,
                              activation='linear'))(geometry_lstm2)

    # Assign inputs and outputs of the model
    if use_context is True:
        geometry_model = Model(input=[prior_geometry_input,
                                      prior_morphology_input,
                                      noise_input],
                               output=[geometry_output])
    else:
        geometry_model = Model(input=[noise_input],
                               output=[geometry_output])

    # ---------------------------
    # Conditional Geometry model
    # ---------------------------

    # Concatenate common inputs with specific input
    all_geometry_inputs = merge([all_common_inputs,
                                 morphology_embedding])

    # Dense
    geometry_hidden_dim = (n_nodes_out - 1) * 3
    geometry_hidden1 = Dense(geometry_hidden_dim)(all_geometry_inputs)
    geometry_hidden2 = Dense(geometry_hidden_dim)(geometry_hidden1)

    # Reshape
    geometry_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, 3))(geometry_hidden2)

    # LSTM
    geometry_lstm1 = \
        LSTM(input_dim=3,
             input_length=n_nodes_out - 1,
             output_dim=3,
             return_sequences=True)(geometry_reshaped)
    geometry_lstm2 = \
        LSTM(input_dim=3,
             input_length=n_nodes_out - 1,
             output_dim=3,
             return_sequences=True)(geometry_lstm1)
    # TimeDistributed
    geometry_output = \
        TimeDistributed(Dense(input_dim=3,
                              output_dim=3,
                              activation='linear'))(geometry_lstm2)

    # Assign inputs and outputs of the model
    if use_context is True:
        conditional_geometry_model = \
            Model(input=[prior_geometry_input,
                         prior_morphology_input,
                         noise_input,
                         morphology_input],
                  output=[geometry_output])
    else:
        conditional_geometry_model = \
            Model(input=[noise_input,
                         morphology_input],
                  output=[geometry_output])

    # -----------------
    # Morphology model
    # -----------------

    # Dense
    morphology_hidden_dim = hidden_dim * (n_nodes_out - 1)
    morphology_hidden1 = Dense(morphology_hidden_dim)(all_common_inputs)
    morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, hidden_dim))(morphology_hidden2)

    # LSTM
    morphology_lstm1 = \
        LSTM(input_dim=hidden_dim,
             input_length=n_nodes_out - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_reshaped)
    morphology_lstm2 = \
        LSTM(input_dim=hidden_dim,
             input_length=n_nodes_out - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_lstm1)
    # TimeDistributed
    morphology_dense = \
        TimeDistributed(Dense(input_dim=hidden_dim,
                              output_dim=n_nodes_out,
                              activation='sigmoid'))(morphology_lstm2)

    lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    morphology_output = \
        Lambda(masked_softmax_full,
               output_shape=(n_nodes_out - 1, n_nodes_out),
               arguments=lambda_args)(morphology_dense)

    # # Dense
    # morphology_hidden_dim = n_nodes_out * (n_nodes_out - 1)
    # morphology_hidden1 = Dense(morphology_hidden_dim,
    #                            activation='sigmoid')(all_common_inputs)
    #
    # # Reshape
    # morphology_reshaped = \
    #     Reshape(target_shape=(n_nodes_out - 1, n_nodes_out))(morphology_hidden1)
    #
    # lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    # morphology_output = \
    #     Lambda(masked_softmax_full,
    #            output_shape=(n_nodes_out - 1, n_nodes_out),
    #            arguments=lambda_args)(morphology_reshaped)

    # Assign inputs and outputs of the model
    if use_context is True:
        morphology_model = \
            Model(input=[prior_geometry_input,
                         prior_morphology_input,
                         noise_input],
                  output=[morphology_output])
    else:
        morphology_model = \
            Model(input=[noise_input],
                  output=[morphology_output])

    # -----------------------------
    # Conditional morphology model
    # -----------------------------

    # Concatenate common inputs with specific input
    all_morphology_inputs = merge([all_common_inputs,
                                   geometry_embedding])

    # Dense
    morphology_hidden_dim = hidden_dim * (n_nodes_out - 1)
    morphology_hidden1 = Dense(morphology_hidden_dim)(all_morphology_inputs)
    morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, hidden_dim))(morphology_hidden2)

    # LSTM
    morphology_lstm1 = \
        LSTM(input_dim=hidden_dim,
             input_length=n_nodes_out - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_reshaped)
    morphology_lstm2 = \
        LSTM(input_dim=hidden_dim,
             input_length=n_nodes_out - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_lstm1)

    # TimeDistributed
    morphology_dense = \
        TimeDistributed(Dense(input_dim=hidden_dim,
                              output_dim=n_nodes_out,
                              activation='sigmoid'))(morphology_lstm2)

    lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    morphology_output = \
        Lambda(masked_softmax_full,
               output_shape=(n_nodes_out - 1, n_nodes_out),
               arguments=lambda_args)(morphology_dense)

    # # Dense
    # morphology_hidden_dim = n_nodes_out * (n_nodes_out - 1)
    # morphology_hidden1 = Dense(morphology_hidden_dim,
    #                            activation='softmax')(all_morphology_inputs)
    #
    # # Reshape
    # morphology_reshaped = \
    #     Reshape(target_shape=(n_nodes_out - 1, n_nodes_out))(morphology_hidden1)
    #
    # lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    # morphology_output = \
    #     Lambda(masked_softmax_full,
    #            output_shape=(n_nodes_out - 1, n_nodes_out),
    #            arguments=lambda_args)(morphology_reshaped)

    # Assign inputs and outputs of the model
    if use_context is True:
        conditional_morphology_model = \
            Model(input=[prior_geometry_input,
                         prior_morphology_input,
                         noise_input,
                         geometry_input],
                  output=[morphology_output])
    else:
        conditional_morphology_model = \
            Model(input=[noise_input,
                         geometry_input],
                  output=[morphology_output])

    geometry_model.summary()
    conditional_geometry_model.summary()
    morphology_model.summary()
    conditional_morphology_model.summary()

    return geometry_model, \
        conditional_geometry_model, \
        morphology_model, \
        conditional_morphology_model


# Discriminator
def discriminator(n_nodes_in=10,
                  embedding_dim=100,
                  hidden_dim=50,
                  train_loss='wasserstein_loss'):
    """
    Discriminator network.

    Parameters
    ----------
    n_nodes_in: int
        number of nodes in the tree providing context input
    embedding_dim: int
        dimensionality of embedding for context input
    hidden_dim: int
        dimensionality of hidden layers

    Returns
    -------
    discriminator_model: keras model object
        model of discriminator
    """
    # Joint embedding of geometry and morphology
    geometry_input, morphology_input, embedding = \
        embedder(n_nodes=n_nodes_in,
                 embedding_dim=embedding_dim)

    # --------------------
    # Discriminator model
    # -------------------=
    discriminator_hidden1 = \
        Dense(hidden_dim)(embedding)
    discriminator_hidden2 = \
        Dense(hidden_dim)(discriminator_hidden1)
    if train_loss == 'wasserstein_loss':
        discriminator_output = \
            Dense(1, activation='linear')(discriminator_hidden2)
    else:
        discriminator_output = \
            Dense(1, activation='sigmoid')(discriminator_hidden2)

    discriminator_model = Model(input=[geometry_input,
                                       morphology_input],
                                output=[discriminator_output])

    discriminator_model.summary()
    return discriminator_model


def wasserstein_loss(y_true, y_pred):
    """
    Custom loss function for Wasserstein critic.

    Parameters
    ----------
    y_true: keras tensor
        true labels: -1 for data and +1 for generated sample
    y_pred: keras tensor
        predicted EM score
    """
    return K.mean(y_true * y_pred)


# Discriminator on generators
def discriminator_on_generators(geometry_model,
                                conditional_geometry_model,
                                morphology_model,
                                conditional_morphology_model,
                                discriminator_model,
                                conditioning_rule='mgd',
                                input_dim=100,
                                n_nodes_in=10,
                                n_nodes_out=20,
                                use_context=True):
    """
    Discriminator stacked on the generators.

    Parameters
    ----------
    geometry_model: keras model object
        model object that generates the geometry
    conditional_geometry_model: keras model object
        model object that generates the geometry conditioned on morphology
    morphology_model: keras model object
        model object that generates the morphology
    conditional_morphology_model: keras model object
        model object that generates the morphology conditioned on geometry
    discriminator_model: keras model object
        model object for the discriminator
    conditioning_rule: str
        'mgd': P_w(disc_loss|g,m) P(g|m) P(m)
        'gmd': P_w(disc_loss|g,m) P(m|g) P(g)
    input_dim: int
        dimensionality of noise input
    n_nodes_in: int
        number of nodes in the tree providing
        prior context input for the generators
    n_nodes_out: int
        number of nodes in the output tree
    use_context: bool
        if True, use context, else only noise input for the generators

    Returns
    -------
    model: keras model object
        model of the discriminator stacked on the generator.
    """
    # Inputs
    if use_context is True:
        prior_geometry_input = Input(shape=(n_nodes_in - 1, 3))
        prior_morphology_input = Input(shape=(n_nodes_in - 1, n_nodes_in))

    noise_input = Input(shape=(1, input_dim), name='noise_input')

    prior_geometry_input = Input(shape=(n_nodes_out - 1, 3))
    prior_morphology_input = Input(shape=(n_nodes_out - 1, n_nodes_in))

    # ------------------
    # Generator outputs
    # ------------------
    if conditioning_rule == 'mgd':
        # Condition geometry on morphology: P(g|m)P(m)
        if use_context is True:
            morphology_output = \
                morphology_model([prior_geometry_input,
                                  prior_morphology_input,
                                  noise_input])
            geometry_output = \
                conditional_geometry_model([prior_geometry_input,
                                            prior_morphology_input,
                                            noise_input,
                                            morphology_output])
        else:
            morphology_output = \
                morphology_model([noise_input])
            geometry_output = \
                conditional_geometry_model([noise_input,
                                            morphology_output])

    elif conditioning_rule == 'gmd':
        # Condition morphology on geometry: P(m|g)P(g)
        if use_context is True:
            geometry_output = \
                geometry_model([prior_geometry_input,
                                prior_morphology_input,
                                noise_input])
            morphology_output = \
                conditional_morphology_model([prior_geometry_input,
                                              prior_morphology_input,
                                              noise_input,
                                              geometry_output])
        else:
            geometry_output = \
                morphology_model([noise_input])
            morphology_output = \
                conditional_morphology_model([noise_input,
                                              geometry_output])

    # ---------------------
    # Discriminator output
    # ---------------------
    discriminator_output = \
        discriminator_model([geometry_output,
                             morphology_output])

    # Stack discriminator on generator
    if use_context is True:
        model = Model([prior_geometry_input,
                       prior_morphology_input,
                       noise_input],
                      [discriminator_output])
    else:
        model = Model([noise_input],
                      [discriminator_output])

    return model
