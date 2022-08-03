import tensorflow as tf
from tensorflow.keras import layers


class GaussianSampling(layers.Layer):
    '''
    Draw a sample from a gaussian distribution
    '''

    def call(self, mean, log_variance):
        # batch: the size of the images we are passing on the batch
        batch = tf.shape(mean)[0]

        # dim: the size of the latent space z
        dim = tf.shape(mean)[1]

        # each element of epsilon: a sample of the current gaussian distribution represented by z_mean and z_log_var
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # For 'out' computation:
        # Note 1: We can sample from a multivariate gaussian distribution
        # X = [X0, X1, ..., X_{k-1}]^T (a column vector) ~ N(mean, variance) as follows:
        #
        #             sample =  mean + variance * ep
        # where:
        #   ep is generated from the multivariate standard gaussian distribution N(0, I)
        #   shape of mean  = (k, 1)
        #   shape of variance = (k, k)
        #   shape of ep = (k, 1)
        #   of course, shape of sample  = (k, 1)
        #
        # Note 2: tf.exp(0.5 * z_log_var) is variance
        #
        # Note 3: log means log_e
        out = mean + tf.exp(0.5 * log_variance) * epsilon
        return out
