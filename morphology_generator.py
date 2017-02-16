"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras import backend as K


def masked_softmax(input_layer, n_nodes, batch_size):

    mask_lower = K.theano.tensor.tril(K.ones((n_nodes - 1, n_nodes)))
    mask_upper = \
        K.theano.tensor.triu(-100. * K.ones((n_nodes - 1, n_nodes)), 1)
    mask_layer = mask_lower * input_layer + mask_upper
    mask_layer = mask_layer + 2*K.eye(n_nodes)[0:n_nodes - 1,0:n_nodes]
    mask_layer = \
        K.reshape(mask_layer, (batch_size * (n_nodes - 1), n_nodes))
    softmax_layer = K.softmax(mask_layer)
    output_layer = K.reshape(softmax_layer, (batch_size, n_nodes - 1, n_nodes))
    return output_layer

def full_matrix(adjacency, n_nodes):
    return K.theano.tensor.nlinalg.matrix_inverse(K.eye(n_nodes) - adjacency)


# Masked softmax Lambda layer
def masked_softmax_full(input_layer, n_nodes, batch_size):
    mask_layer = masked_softmax(input_layer, n_nodes, batch_size)
    mask_layer = \
        K.concatenate([K.zeros(shape=[batch_size, 1, n_nodes]), mask_layer],
                      axis=1)
    result, updates = \
        K.theano.scan(fn=lambda n: full_matrix(mask_layer[n, : , :], n_nodes),
                      sequences=K.arange(batch_size))
    return result[:, 1:, :]


def generator(n_nodes_in=10,
              n_nodes_out=20,
              noise_dim=100,
              embedding_dim=100,
              hidden_dim=30,
              batch_size=64,
              use_context=True):

    noise_input = Input(shape=(1, noise_dim), name='noise_input')
    #hidden_dim = n_nodes_out
    morphology_hidden_dim = hidden_dim * (n_nodes_out - 1)
    morphology_hidden1 = Dense(morphology_hidden_dim)(noise_input)
    morphology_hidden2 = Dense(morphology_hidden_dim)(morphology_hidden1)
    morphology_hidden3 = Dense(n_nodes_out * (n_nodes_out - 1))(morphology_hidden2)
    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes_out - 1, n_nodes_out))(morphology_hidden3)
    # # LSTM
    # morphology_lstm1 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          output_dim=hidden_dim,
    #          return_sequences=True)(morphology_reshaped)
    # morphology_lstm2 = \
    #     LSTM(input_dim=hidden_dim,
    #          input_length=n_nodes_out - 1,
    #          output_dim=hidden_dim,
    #          return_sequences=True)(morphology_lstm1)
    # # TimeDistributed
    # morphology_dense = \
    #     TimeDistributed(Dense(input_dim=hidden_dim,
    #                           output_dim=n_nodes_out,
    #                           activation='sigmoid'))(morphology_lstm2)

    lambda_args = {'n_nodes': n_nodes_out, 'batch_size': batch_size}
    morphology_output = \
        Lambda(masked_softmax_full,
               output_shape=(n_nodes_out - 1, n_nodes_out),
               arguments=lambda_args)(morphology_reshaped)

    # Assign inputs and outputs of the model
    morphology_model = \
        Model(input=[noise_input],
              output=[morphology_output])

    morphology_model.summary()
    return morphology_model


# Discriminator
def discriminator(n_nodes=10,
                  embedding_dim=100,
                  hidden_dim=50,
                  train_loss='wasserstein_loss'):
    morphology_input = Input(shape=(n_nodes - 1, n_nodes))

    # LSTM
    embedding_lstm1 = \
        LSTM(input_dim=(n_nodes),
             input_length=n_nodes - 1,
             output_dim=hidden_dim,
             return_sequences=True)(morphology_input)

    embedding_reshaped = \
        Reshape(target_shape=
                (1, (n_nodes - 1) * hidden_dim))(embedding_lstm1)

    embedding = Dense(input_dim=(n_nodes - 1) * hidden_dim,
                      output_dim=embedding_dim,
                      name='embedding')(embedding_reshaped)
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

    discriminator_model = Model(input=[morphology_input],
                                output=[discriminator_output])

    discriminator_model.summary()
    return discriminator_model


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def discriminator_on_generators(morphology_model,
                                discriminator_model,
                                conditioning_rule='mgd',
                                input_dim=100,
                                n_nodes_in=10,
                                n_nodes_out=20,
                                use_context=True):


    noise_input = Input(shape=(1, input_dim), name='noise_input')

    if conditioning_rule == 'm':
        morphology_output = \
            morphology_model([noise_input])

    discriminator_output = \
        discriminator_model([morphology_output])

    model = Model([noise_input],
                  [discriminator_output])
    return model
