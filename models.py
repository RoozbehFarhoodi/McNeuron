"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.layers.core import Dense, Reshape, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras.constraints import maxnorm


# Embedding layers
def geometry_embedder(n_nodes=10, embedding_dim=100):
    """
    Embedding of geometric coordinates of nodes.

    Parameters
    ----------
    n_nodes: int
        number of nodes
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
    geometry_reshaped = \
        Reshape(target_shape=(1, (n_nodes - 1) * 3))(geometry_input)
    geometry_embedding = Dense(input_dim=(n_nodes - 1) * 3,
                               output_dim=embedding_dim,
                               name='geometry_embedding')(geometry_reshaped)
    return geometry_input, geometry_embedding


def morphology_embedder(n_nodes=10, embedding_dim=100):
    """
    Embedding of tree morphology (softmax Prufer code).

    Parameters
    ----------
    n_nodes: int
        number of nodes
    embedding_dim: int
        embedding_dimension

    Returns
    -------
    morphology_input: keras layer object
        input layer
    morphology_embedding: keras layer object
        embedding layer
    """
    morphology_input = Input(shape=(n_nodes - 2, n_nodes))
    morphology_reshaped = \
        Reshape(target_shape=(1, n_nodes * (n_nodes - 2)))(morphology_input)
    morphology_embedding = \
        Dense(input_dim=(n_nodes - 2) * n_nodes,
              output_dim=embedding_dim,
              name='morphology_embedding')(morphology_reshaped)
    return morphology_input, morphology_embedding


# Generators
def generator(n_nodes_in=10,
              n_nodes_out=20,
              noise_dim=100,
              embedding_dim=100,
              hidden_dim=10,
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
    morphology_model: keras model object
        model of morphology generator
    """
    # Embed contextual information
    if use_context is True:
        geometry_input, geometry_embedding = \
            geometry_embedder(n_nodes=n_nodes_in,
                              embedding_dim=embedding_dim)
        morphology_input, morphology_embedding = \
            morphology_embedder(n_nodes=n_nodes_in,
                                embedding_dim=embedding_dim)

    # Generate noise input
    noise_input = Input(shape=(1, noise_dim), name='noise_input')

    # Concatenate context and noise inputs
    if use_context is True:
        all_inputs = merge([geometry_embedding,
                            morphology_embedding,
                            noise_input], mode='concat')
    else:
        all_inputs = noise_input

    # -------------------
    # Geometry model
    # -------------------
    # Synthesize output
    geometry_hidden_dim = (n_nodes_out - 1) * 3
    geometry_hidden = Dense(geometry_hidden_dim)(all_inputs)

    # Reshape
    geometry_output = \
        Reshape(target_shape=(n_nodes_out - 1, 3))(geometry_hidden)

    # Assign inputs and outputs of the model
    if use_context is True:
        geometry_model = Model(input=[geometry_input,
                                      morphology_input,
                                      noise_input],
                               output=[geometry_output])
    else:
        geometry_model = Model(input=[noise_input],
                               output=[geometry_output])

    # -------------------
    # Morphology model
    # -------------------
    # Synthesize output
    morphology_hidden_dim = hidden_dim * (n_nodes_out - 2)
    morphology_hidden = Dense(morphology_hidden_dim)(all_inputs)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 2, hidden_dim))(morphology_hidden)

    # LSTM
    morphology_lstm = \
        LSTM(input_dim=hidden_dim,
             input_length=n_nodes_out - 2,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_reshaped)

    # TimeDistributed
    morphology_output = \
        TimeDistributed(Dense(input_dim=hidden_dim,
                              output_dim=n_nodes_out,
                              activation='softmax'))(morphology_lstm)

    # Assign inputs and outputs of the model
    if use_context is True:
        morphology_model = \
            Model(input=[geometry_input,
                         morphology_input,
                         noise_input],
                  output=[morphology_output])
    else:
        morphology_model = Model(input=[noise_input],
                                 output=[morphology_output])

    return geometry_model, morphology_model


# Discriminator
def discriminator(n_nodes_in=10,
                  embedding_dim=100,
                  hidden_dim=50):
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
    # Embed geometry
    geometry_input, geometry_embedding = \
        geometry_embedder(n_nodes=n_nodes_in,
                          embedding_dim=embedding_dim)

    # Embed morphology
    morphology_input, morphology_embedding = \
        morphology_embedder(n_nodes=n_nodes_in,
                            embedding_dim=embedding_dim)

    # Concatenate embeddings
    all_inputs = merge([geometry_embedding, morphology_embedding],
                       mode='concat')

    # --------------------
    # Discriminator model
    # -------------------=
    discriminator_hidden1 = \
        Dense(hidden_dim)(all_inputs)
    discriminator_hidden2 = \
        Dense(hidden_dim)(discriminator_hidden1)
    discriminator_output = \
        Dense(1, activation='linear')(discriminator_hidden2)

    discriminator_model = Model(input=[geometry_input,
                                       morphology_input],
                                output=[discriminator_output])

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
                                morphology_model,
                                discriminator_model,
                                input_dim=100,
                                n_nodes_in=10,
                                use_context=True):
    """
    Discriminator stacked on the generators.

    Parameters
    ----------
    geometry_model: keras model object
        model object that generates the geometry
    morphology_model: keras model object
        model object that generates the morphology
    discriminator_model: keras model object
        model object for the discriminator
    input_dim: int
        dimensionality of noise input
    n_nodes_in: int
        number of nodes in the tree providing
        context input for the generators
    use_context: bool
        if True, use context, else only noise input for the generators

    Returns
    -------
    gan_model: keras model object
        model of the discriminator stacked on the generator
    """
    # Inputs
    if use_context is True:
        geometry_input = Input(shape=(n_nodes_in - 1, 3))
        morphology_input = Input(shape=(n_nodes_in - 2, n_nodes_in))

    noise_input = Input(shape=(1, input_dim), name='noise_input')

    # Generator outputs
    if use_context is True:
        geometry_output = geometry_model([geometry_input,
                                          morphology_input,
                                          noise_input])
        morphology_output = morphology_model([geometry_input,
                                              morphology_input,
                                              noise_input])
    else:
        geometry_output = geometry_model(noise_input)
        morphology_output = morphology_model(noise_input)

    # Discriminator output
    discriminator_output = \
        discriminator_model([geometry_output,
                             morphology_output])

    # Stack discriminator on generator
    if use_context is True:
        gan_model = Model([geometry_input,
                           morphology_input,
                           noise_input],
                          [discriminator_output])
    else:
        gan_model = Model([noise_input],
                          [discriminator_output])
    return gan_model
