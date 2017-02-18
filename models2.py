"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

# Local imports
import layers as layers


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
        prior_geometry_input = \
            Input(shape=(n_nodes_in - 1, 3))
        prior_morphology_input = \
            Input(shape=(n_nodes_in - 1, n_nodes_in))

        # prior_embedding = \
        #     layers.embedder(prior_geometry_input,
        #                     prior_morphology_input,
        #                     n_nodes=n_nodes_in,
        #                     hidden_dim=hidden_dim,
        #                     embedding_dim=embedding_dim)
        lambda_args = {'n_nodes': n_nodes_in, 'batch_size': batch_size}
        prior_embedding = \
            Lambda(layers.feature_extractor,
                   output_shape=(n_nodes_in, 2 * n_nodes_in + 3),
                   arguments=lambda_args)([prior_geometry_input,
                                           prior_morphology_input])
        prior_embedding = \
            Reshape(target_shape=(1, n_nodes_in * (2 * n_nodes_in + 3)))(prior_embedding)
    # Generate noise input
    noise_input = Input(shape=(1, noise_dim), name='noise_input')

    # Geometry and morphology input
    geometry_input = Input(shape=(n_nodes_out - 1, 3))
    morphology_input = Input(shape=(n_nodes_out - 1, n_nodes_out))

    # Concatenate prior context and noise inputs
    if use_context is True:
        all_common_inputs = merge([prior_embedding,
                                   noise_input], mode='concat')
    else:
        all_common_inputs = noise_input

    # Embed conditional information from current level
    geometry_embedding = \
        layers.geometry_embedder(geometry_input,
                                 n_nodes=n_nodes_out,
                                 hidden_dim=hidden_dim,
                                 embedding_dim=embedding_dim)
    morphology_embedding = \
        layers.morphology_embedder(morphology_input,
                                   n_nodes=n_nodes_out,
                                   hidden_dim=hidden_dim,
                                   embedding_dim=embedding_dim)

    # ---------------
    # Geometry model
    # ---------------

    # Dense
    geometry_hidden_dim = (n_nodes_out - 1) * 3
    geometry_hidden1 = Dense(geometry_hidden_dim)(all_common_inputs)
    # geometry_hidden1 = BatchNormalization()(geometry_hidden1)
    geometry_hidden2 = Dense(geometry_hidden_dim)(geometry_hidden1)
    # geometry_hidden2 = BatchNormalization()(geometry_hidden2)

    # Reshape
    geometry_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, 3))(geometry_hidden2)

    # # LSTM
    # geometry_lstm1 = \
    #     LSTM(input_dim=3,
    #          input_length=n_nodes_out - 1,
    #          output_dim=3,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(geometry_reshaped)
    # # geometry_lstm1 = BatchNormalization()(geometry_lstm1)
    #
    # geometry_lstm2 = \
    #     LSTM(input_dim=3,
    #          input_length=n_nodes_out - 1,
    #          output_dim=3,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(geometry_lstm1)
    # # geometry_lstm2 = BatchNormalization()(geometry_lstm2)
    #
    # # TimeDistributed
    # geometry_output = \
    #     TimeDistributed(Dense(input_dim=3,
    #                           output_dim=3,
    #                           W_regularizer=l2(0.01),
    #                           activation='linear'))(geometry_lstm2)
    geometry_output = geometry_reshaped

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
    # geometry_hidden1 = BatchNormalization()(geometry_hidden1)
    geometry_hidden2 = Dense(geometry_hidden_dim)(geometry_hidden1)
    # geometry_hidden2 = BatchNormalization()(geometry_hidden2)

    # Reshape
    geometry_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, 3))(geometry_hidden2)

    # # LSTM
    # geometry_lstm1 = \
    #     LSTM(input_dim=3,
    #          input_length=n_nodes_out - 1,
    #          output_dim=3,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(geometry_reshaped)
    # # geometry_lstm1 = BatchNormalization()(geometry_lstm1)
    #
    # geometry_lstm2 = \
    #     LSTM(input_dim=3,
    #          input_length=n_nodes_out - 1,
    #          output_dim=3,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(geometry_lstm1)
    # # geometry_lstm2 = BatchNormalization()(geometry_lstm2)
    #
    # # TimeDistributed
    # geometry_output = \
    #     TimeDistributed(Dense(input_dim=3,
    #                           output_dim=3,
    #                           W_regularizer=l2(0.01),
    #                           activation='linear'))(geometry_lstm2)

    geometry_output = geometry_reshaped

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

    # # Dense
    # morphology_hidden_dim = hidden_dim * (n_nodes_out - 1)
    # morphology_hidden1 = Dense(morphology_hidden_dim)(all_common_inputs)
    # morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)
    #
    # # Reshape
    # morphology_reshaped = \
    #     Reshape(target_shape=(n_nodes_out - 1, hidden_dim))(morphology_hidden2)
    #
    # # LSTM
    # morphology_lstm1 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          output_dim=hidden_dim,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(morphology_reshaped)
    # morphology_lstm2 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          output_dim=hidden_dim,
    #          return_sequences=True)(morphology_lstm1)
    # # TimeDistributed
    # morphology_dense = \
    #     TimeDistributed(Dense(input_dim=hidden_dim,
    #                           output_dim=n_nodes_out,
    #                           W_regularizer=l2(0.01),
    #                           activation='sigmoid'))(morphology_lstm1)
    #
    # lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    # morphology_output = \
    #     Lambda(layers.masked_softmax,
    #            output_shape=(n_nodes_out - 1, n_nodes_out),
    #            arguments=lambda_args)(morphology_dense)

    # Dense
    morphology_hidden_dim = n_nodes_out * (n_nodes_out - 1)
    morphology_hidden1 = Dense(morphology_hidden_dim)(all_common_inputs)
    # morphology_hidden1 = BatchNormalization()(morphology_hidden1)
    morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)
    # morphology_hidden2 = BatchNormalization()(morphology_hidden2)
    morphology_hidden3 = Dense(n_nodes_out * (n_nodes_out - 1),
                               activation='linear')(morphology_hidden2)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, n_nodes_out))(morphology_hidden3)

    lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    morphology_output = \
        Lambda(layers.masked_softmax_full,
               output_shape=(n_nodes_out - 1, n_nodes_out),
               arguments=lambda_args)(morphology_reshaped)

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

    # # Dense
    # morphology_hidden_dim = hidden_dim * (n_nodes_out - 1)
    # morphology_hidden1 = Dense(morphology_hidden_dim)(all_morphology_inputs)
    # morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)
    #
    # # Reshape
    # morphology_reshaped = \
    #     Reshape(target_shape=(n_nodes_out - 1, hidden_dim))(morphology_hidden2)
    #
    # # LSTM
    # morphology_lstm1 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          output_dim=hidden_dim,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(morphology_reshaped)
    # morphology_lstm2 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          output_dim=hidden_dim,
    #          W_regularizer=l2(0.1),
    #          U_regularizer=l2(0.1),
    #          return_sequences=True)(morphology_lstm1)
    #
    # # TimeDistributed
    # morphology_dense = \
    #     TimeDistributed(Dense(input_dim=hidden_dim,
    #                           output_dim=n_nodes_out,
    #                           W_regularizer=l2(0.01),
    #                           activation='sigmoid'))(morphology_lstm1)
    #
    # lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    # morphology_output = \
    #     Lambda(layers.masked_softmax,
    #            output_shape=(n_nodes_out - 1, n_nodes_out),
    #            arguments=lambda_args)(morphology_dense)

    # Dense
    morphology_hidden_dim = n_nodes_out * (n_nodes_out - 1)
    morphology_hidden1 = Dense(morphology_hidden_dim)(all_morphology_inputs)
    # morphology_hidden1 = BatchNormalization()(morphology_hidden1)
    morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)
    # morphology_hidden2 = BatchNormalization()(morphology_hidden2)
    morphology_hidden3 = Dense(n_nodes_out * (n_nodes_out - 1),
                               activation='linear')(morphology_hidden2)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, n_nodes_out))(morphology_hidden1)

    lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    morphology_output = \
        Lambda(layers.masked_softmax_full,
               output_shape=(n_nodes_out - 1, n_nodes_out),
               arguments=lambda_args)(morphology_reshaped)

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

    # geometry_model.summary()
    conditional_geometry_model.summary()
    morphology_model.summary()
    # conditional_morphology_model.summary()

    return geometry_model, \
        conditional_geometry_model, \
        morphology_model, \
        conditional_morphology_model


# Discriminator
def discriminator(n_nodes_in=10,
                  embedding_dim=100,
                  hidden_dim=50,
                  batch_size=64,
                  train_loss='wasserstein_loss'):
    """
    Discriminator network.

    Parameters
    ----------
    n_nodes_in: int
        number of nodes in the tree
    embedding_dim: int
        dimensionality of embedding for context input
    hidden_dim: int
        dimensionality of hidden layers

    Returns
    -------
    discriminator_model: keras model object
        model of discriminator
    """
    geometry_input = Input(shape=(n_nodes_in - 1, 3))
    morphology_input = Input(shape=(n_nodes_in - 1, n_nodes_in))

    # # Joint embedding of geometry and morphology
    # embedding = layers.embedder(geometry_input,
    #                             morphology_input,
    #                             n_nodes=n_nodes_in,
    #                             embedding_dim=embedding_dim)

    # Extract features from geometry and morphology
    lambda_args = {'n_nodes': n_nodes_in, 'batch_size': 2 * batch_size}
    embedding = \
        Lambda(layers.feature_extractor,
               output_shape=(n_nodes_in, 2 * n_nodes_in + 3),
               arguments=lambda_args)([geometry_input,
                                       morphology_input])
    embedding = \
        Reshape(target_shape=(1, n_nodes_in * (2 * n_nodes_in + 3)))(embedding)

    # --------------------
    # Discriminator model
    # -------------------=
    discriminator_hidden1 = Dense(hidden_dim)(embedding)
    # discriminator_hidden1 = Dropout(0.1)(discriminator_hidden1)
    discriminator_hidden2 = Dense(hidden_dim)(discriminator_hidden1)
    # discriminator_hidden2 = Dropout(0.1)(discriminator_hidden2)
    discriminator_hidden3 = Dense(hidden_dim)(discriminator_hidden2)
    # discriminator_hidden3 = Dropout(0.1)(discriminator_hidden3)

    if train_loss == 'wasserstein_loss':
        discriminator_output = \
            Dense(1, activation='linear')(discriminator_hidden3)
    else:
        discriminator_output = \
            Dense(1, activation='sigmoid')(discriminator_hidden3)

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

    # prior_geometry_input = Input(shape=(n_nodes_out - 1, 3))
    # prior_morphology_input = Input(shape=(n_nodes_out - 1, n_nodes_in))

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
                geometry_model([noise_input])
            morphology_output = \
                conditional_morphology_model([noise_input,
                                              geometry_output])

    elif conditioning_rule == 'none':
        # No conditioning
        if use_context is True:
            geometry_output = \
                geometry_model([prior_geometry_input,
                                prior_morphology_input,
                                noise_input])
            morphology_output = \
                morphology_model([prior_geometry_input,
                                  prior_morphology_input,
                                  noise_input])
        else:
            geometry_output = \
                geometry_model([noise_input])
            morphology_output = \
                morphology_model([noise_input])

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
