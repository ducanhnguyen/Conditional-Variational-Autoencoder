import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src.GaussianSampling import GaussianSampling
from utils import mergeTF


class PuVAE(Model):
    def __init__(self, img_input_shape, n_classes, latent_dim=16, res_weight=1, kl_weight=1, ce_weight=1,
                 target_classifier=None, **kwargs):
        super(PuVAE, self).__init__(**kwargs)
        height, width, channels = img_input_shape
        self.target_classifier = target_classifier
        self.res_weight = res_weight
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight

        '''
         ENCODER
         '''
        dense_depth = 0
        conv_depth = 0
        last_dense_shape = None
        last_conv_shape = None

        encoder_inputs = keras.Input(shape=(height + n_classes, width, channels), name="img+one-hot class")
        x = encoder_inputs

        while True:
            if x.shape[1] < 10 or x.shape[2] < 10:  # height or width
                break

            conv_depth += 1
            # *2 the number of channels while /2 the size of feature maps
            x = layers.Conv2D(filters=x.shape[3] * 4,  # * 2 the number of channels
                              kernel_size=3,
                              activation="relu",
                              strides=2,  # /2 width and height of feature maps
                              padding="same"
                              )(x)
            last_conv_shape = x.shape

        x = layers.Flatten()(x)

        while True:
            if x.shape[1] < 256 or x.shape[1] < latent_dim * 2:
                break
            x = layers.Dense(x.shape[1] / 2, activation="relu")(x)
            dense_depth += 1
            last_dense_shape = x.shape

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        z = GaussianSampling()(mean=z_mean,
                               log_variance=z_log_var)

        encoder = keras.Model(inputs=encoder_inputs,
                              outputs=[z_mean, z_log_var, z],
                              name="encoder")
        encoder.summary()

        '''
        DECODER
        '''
        latent_inputs = keras.Input(shape=(latent_dim + n_classes,))

        x = layers.Dense(last_dense_shape[1], activation="relu")(latent_inputs)

        while dense_depth > 0:
            x = layers.Dense(x.shape[1] * 2, activation="relu")(x)
            dense_depth -= 1

        x = layers.Reshape((last_conv_shape[1], last_conv_shape[2], last_conv_shape[3]))(x)

        while conv_depth > 0:
            x = layers.Conv2DTranspose(filters=x.shape[3] / 4,
                                       kernel_size=3,
                                       activation="relu",
                                       strides=2,
                                       padding="same")(x)
            conv_depth -= 1

        x = layers.Conv2D(filters=x.shape[3],
                          kernel_size=(x.shape[1] - height + 1, 1),
                          activation="sigmoid",
                          strides=1,
                          name="last"
                          )(x)

        decoder = keras.Model(inputs=latent_inputs,
                              outputs=x,
                              name="decoder")
        decoder.summary()

        '''
        MERGE
        '''
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.ce_loss_tracker = keras.metrics.Mean(
            name="ce_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
            ,
            self.ce_loss_tracker
        ]

    def call(self, inputs):
        images, one_hot_labels = inputs
        merged_inputs = mergeTF(images, one_hot_labels)
        _mean, z_log_var, z = self.encoder(merged_inputs)
        z = tf.concat((z, one_hot_labels), axis=1)
        return self.decoder(z)

    @tf.function
    def test(self, model, x):
        y = model.inference(x, steps=1)

        return y

    def train_step(self, data):
        images, one_hot_labels = data[0]
        merged_inputs = mergeTF(images, one_hot_labels)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(merged_inputs)

            # shape of z = (batch, latent_dim)
            z = tf.concat((z, one_hot_labels), axis=1)  # (None, len(z) + len(one_hot_labels))
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)
                )
            )
            reconstruction_loss = self.res_weight * reconstruction_loss

            kl_loss = 0.5 * (tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = self.kl_weight * kl_loss

            ce_loss = tf.reduce_mean(
                tf.keras.metrics.categorical_crossentropy(one_hot_labels,
                                                          self.target_classifier(reconstruction))
            )
            ce_loss = self.ce_weight * ce_loss
            total_loss = reconstruction_loss + kl_loss + ce_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result()
        }