"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Dropout, Activation
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM


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
    geom_input: keras layer object
        input layer
    geom_embedded: keras layer object
        embedding layer
    """
    geom_input = Input(shape=(n_nodes - 1, 3))
    geom_reshaped = Reshape(target_shape=(1, (n_nodes - 1) * 3))(geom_input)
    geom_embedded = Dense(input_dim=(n_nodes - 1) * 3,
                          output_dim=embedding_dim,
                          name='geom_embed')(geom_reshaped)
    return geom_input, geom_embedded


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
    morph_input: keras layer object
        input layer
    morph_embedded: keras layer object
        embedding layer
    """
    morph_input = Input(shape=(n_nodes - 2, n_nodes))
    morph_reshaped = \
        Reshape(target_shape=(1, (n_nodes - 2) * n_nodes))(morph_input)
    morph_embedded = Dense(input_dim=(n_nodes - 2) * n_nodes,
                           output_dim=embedding_dim,
                           name='morph_embed')(morph_reshaped)
    return morph_input, morph_embedded


# Generators
def generator(n_nodes_in=10,
              n_nodes_out=20,
              input_size=100,
              embedding_size=100,
              hidden_size=10,
              use_context=True):
    """
    Generator network.

    Parameters
    ----------
    n_nodes_in: int
        number of nodes in the neuron providing context input
    n_nodes_out: int
        number of nodes in the output neuron
    input_size: int
        dimensionality of noise input
    embedding_size: int
        dimensionality of embedding for context input
    use_context: bool
        if True, use context, else only noise input

    Returns
    -------
    geom_model: keras model object
        model of geometry generator
    morph_model: keras model object
        model of
    """
    # Embed contextual information
    if use_context is True:
        geom_input, geom_embedded = \
            geometry_embedder(n_nodes=n_nodes_in,
                              embedding_dim=embedding_size)
        morph_input, morph_embedded = \
            morphology_embedder(n_nodes=n_nodes_in,
                                embedding_dim=embedding_size)

    # Generate noise input
    noise_input = Input(shape=(1, input_size), name='noise_input')

    # Concatenate context and noise inputs
    if use_context is True:
        all_inputs = merge([geom_embedded,
                            morph_embedded,
                            noise_input], mode='concat')
    else:
        all_inputs = noise_input

    # -------------------
    # Geometry model
    # -------------------
    # Synthesize output
    geom_output_size = (n_nodes_out - 1) * 3
    geom_hidden = Dense(geom_output_size)(all_inputs)

    # Reshape
    geom_output = Reshape(target_shape=(n_nodes_out - 1, 3))(geom_hidden)

    # Assign inputs and outputs of the model
    if use_context is True:
        geom_model = Model(input=[geom_input,
                                  morph_input,
                                  noise_input],
                           output=geom_output)
    else:
        geom_model = Model(input=noise_input, output=geom_output)

    # -------------------
    # Morphology model
    # -------------------
    # Synthesize output
    morph_hidden_size = hidden_size * (n_nodes_out - 2)
    morph_hidden = Dense(morph_hidden_size)(all_inputs)

    # Reshape
    morph_reshaped = \
        Reshape(target_shape=(n_nodes_out - 2, hidden_size))(morph_hidden)

    # LSTM
    morph_lstm = LSTM(input_dim=hidden_size,
                      input_length=n_nodes_out - 2,
                      output_dim=hidden_size,
                      return_sequences=True)(morph_reshaped)

    # TimeDistributed
    morph_output = TimeDistributed(Dense(input_dim=hidden_size,
                                         output_dim=n_nodes_out,
                                         activation='softmax'))(morph_lstm)

    # Assign inputs and outputs of the model
    if use_context is True:
        morph_model = \
            Model(input=[geom_input,
                         morph_input,
                         noise_input],
                  output=morph_output)
    else:
        morph_model = Model(input=noise_input,
                            output=morph_output)

    return geom_model, morph_model


def discriminator(n_nodes_in=10,
                  embedding_size=100,
                  hidden_size=50):
    """
    Discriminator network.

    Parameters
    ----------
    n_nodes_in: int
        number of nodes in the neuron providing context input
    embedding_size: int
        dimensionality of embedding for context input
    hidden_size: int
        dimensionality of hidden layers

    Returns
    -------
    disc_model: keras model object
        model of discriminator
    """
    # Embed geometry
    geom_input, geom_embedded = \
        geometry_embedder(n_nodes=n_nodes_in,
                          embedding_dim=embedding_size)

    # Embed morphology
    morph_input, morph_embedded = \
        morphology_embedder(n_nodes=n_nodes_in,
                            embedding_dim=embedding_size)

    # Concatenate embeddings
    all_inputs = merge([geom_embedded, morph_embedded],
                       mode='concat')

    # --------------------
    # Discriminator model
    # -------------------=
    disc_hidden1 = Dense(hidden_size)(all_inputs)
    disc_hidden2 = Dense(hidden_size)(disc_hidden1)
    disc_output = Dense(1, activation='sigmoid')(disc_hidden2)

    disc_model = Model(input=[geom_input,
                              morph_input],
                       output=disc_output)

    return disc_model


def discriminator_on_generators(geom_model,
                                morph_model,
                                disc_model,
                                input_size=100,
                                n_nodes_in=10,
                                use_context=True):
    """
    Discriminator stacked on the generators.

    Parameters
    ----------
    geom_model: keras model object
        model object that generates the geometry
    morph_model: keras model object
        model object that generates the morphology
    disc_model: keras model object
        model object for the discriminator
    n_nodes_in: int
        number of nodes in the neuron providing
        context input for the generators
    embedding_size: int
        dimensionality of embedding for context input
    hidden_size: int
        dimensionality of hidden layers
    use_context: bool
        if True, use context, else only noise input for the generators

    Returns
    -------
    model: keras model object
        model of the discriminator stacked on the generator
    """
    # Inputs
    if use_context is True:
        geom_input = Input(shape=(n_nodes_in - 1, 3))
        morph_input = Input(shape=(n_nodes_in - 2, n_nodes_in))

    noise_input = Input(shape=(1, input_size), name='noise_input')

    # Generator outputs
    if use_context is True:
        geom_output = geom_model([geom_input,
                                  morph_input,
                                  noise_input])
        morph_output = morph_model([geom_input,
                                    morph_input,
                                    noise_input])
    else:
        geom_output = geom_model(noise_input)
        morph_output = morph_model(noise_input)

    # Freeze discriminator
    disc_model.trainable = False

    # Discriminator output
    disc_output = disc_model([geom_output,
                              morph_output])

    # Stack discriminator on generator
    if use_context is True:
        model = Model([geom_input,
                       morph_input,
                       noise_input],
                      disc_output)
    else:
        model = Model(noise_input,
                      disc_output)
    return model
