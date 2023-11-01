import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *


class MLP(keras.layers.Layer):
    """
    A Multi-Layer Perceptron layer.

    This layer is a feedforward neural network composed of multiple dense
    layers, optionally followed by leaky ReLU activations and dropout layers.
    An additional dense layer can be added at the end to produce an output
    of a specified length.

    Attributes
    ----------
    layers : list
        List of keras layers used in the MLP.

    Parameters
    ----------
    n_dense : int, optional
        Number of neurons in the dense layers.
    relu_slope : float, optional
        The slope of the leaky ReLU activation.
    dropout : float, optional
        Fraction of the input units to drop.
    depth : int, optional
        Number of dense layers with leaky ReLU activation to add.
    out_len : int, optional
        Length of the output layer. If specified, an additional dense layer
        is added at the end of the MLP with linear activation.
    """
    def __init__(self, n_dense=128, relu_slope=0.2, dropout=0.2, depth=1, out_len=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        for n in range(int(depth)):
            self.layers.append(Dense(int(n_dense)))
            self.layers.append(LeakyReLU(float(relu_slope)))
            if dropout:
                self.layers.append(Dropout(float(dropout)))
        if out_len is not None:
            self.layers.append(Dense(int(out_len)))

    def call(self, inputs, training=None):
        out = inputs
        for l in self.layers:
            out = l(out, training=training)
        return out


class Sampling(keras.layers.Layer):
    """
    A sampling layer for Variational Autoencoders.

    This layer samples from a Gaussian distribution using the provided
    means and log-variances.

    Returns
    -------
    tensor
        Sampled tensor values.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GaussianEncoder(keras.layers.Layer):
    """
    A Gaussian encoder layer for Variational Autoencoders.

    This layer encodes inputs into a Gaussian distribution in the latent
    space. It consists of an encoder followed by convolutional layers to
    produce the mean and log-variance of the latent variables. The
    reparameterization trick is used to sample from this distribution.

    Attributes
    ----------
    encoder : keras.layers.Layer
        Encoder layer used before producing the Gaussian parameters.
    z_mean : keras.layers.Conv1D
        Convolutional layer to produce the means of the latent variables.
    z_log_var : keras.layers.Conv1D
        Convolutional layer to produce the log-variances of the latent variables.
    sampling : Sampling
        Sampling layer to sample from the Gaussian distribution.

    Parameters
    ----------
    input_len : int
        Length of the input to the encoder.
    latent_len : int
        Length of the latent space.
    encoder : keras.layers.Layer
        Encoder layer to be used in the Gaussian encoder.

    Returns
    -------
    tuple
        Mean, log-variance, and sampled value from the latent distribution.
    """
    def __init__(self, input_len, latent_len, encoder, **kwargs):
        super(GaussianEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.input_len = input_len
        self.latent_len = latent_len
        self.z_mean = Conv1D(filters=int(latent_len), kernel_size=int(input_len), padding='valid')
        self.z_log_var = Conv1D(filters=int(latent_len), kernel_size=int(input_len), padding='valid')
        self.sampling = Sampling()

    def call(self, inputs):
        out = tf.expand_dims(self.encoder(inputs), axis=-1)
        z_mean = tf.squeeze(self.z_mean(out), axis=-2)
        z_log_var = tf.squeeze(self.z_log_var(out), axis=-2)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_len": self.input_len,
            "latent_len": self.latent_len,
        })
        return config
