import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *


class MLP(keras.layers.Layer):
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
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GaussianEncoder(keras.layers.Layer):
    def __init__(self, input_len, latent_len, encoder, **kwargs):
        super(GaussianEncoder, self).__init__(**kwargs)
        self.encoder = encoder
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
