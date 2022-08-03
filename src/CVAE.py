import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model

from src.GaussianSampling import GaussianSampling
from matplotlib import pyplot as plt


class CVAE(Model):
    def __init__(self, img_input_shape, n_classes, latent_dim=16, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        height, width, channels = img_input_shape

        '''
        ENCODER
        '''
        dense_depth = 0
        conv_depth = 0
        last_dense_shape = None
        last_conv_shape = None

        encoder_inputs = keras.Input(shape=(height + n_classes, width, channels))
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
            # if conv_depth == 1:
            #     x = layers.Conv2DTranspose(filters=x.shape[3] / 4,
            #                                kernel_size=3,
            #                                activation="sigmoid",
            #                                strides=2,
            #                                padding="same")(x)
            # else:
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

        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        images, one_hot_labels = inputs
        merged_inputs = mergeTF(images, one_hot_labels)
        _mean, z_log_var, z = self.encoder(merged_inputs)
        z = tf.concat((z, one_hot_labels), axis=1)
        return self.decoder(z)

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

            kl_loss = 0.5 * (tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def mergeTF(x_train, one_hot_y_train):
    _, h, w, c = x_train.shape  # (60000, 28, 28, 1)

    # y_train = tf.keras.utils.to_categorical(y_train)  # (60000, 10)
    # print(y_train.shape)

    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=2)
    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=3)  # (60000, 10, 1, 1)
    # print(f'y_train.shape = {y_train.shape}')

    one_hot_y_train = tf.tile(one_hot_y_train, multiples=[1, 1, w, c])  # (60000, 10, 28, 1)
    # print(f'y_train.shape = {y_train.shape}')

    tmp = tf.concat((x_train, one_hot_y_train), axis=1)  # (60000, 38, 28, 1)
    return tmp


if __name__ == '__main__':
    '''
    LOAD DATA
    '''
    (x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()

    one_hot_y_train = tf.keras.utils.to_categorical(y_train)
    if len(x_train.shape) == 3:  # Ex: (batch, height, width)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    if len(x_test.shape) == 3:  # Ex: (batch, height, width)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    if np.max(x_train) > 1:
        x_train = x_train.astype("float32") / 255

    '''
    TRAIN MODEL
    '''
    MAX_INDEX = 1000
    MODEL_PATH = '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/mnist_cvae'

    x_train = x_train[:MAX_INDEX]
    y_train = y_train[:MAX_INDEX]
    one_hot_y_train = one_hot_y_train[:MAX_INDEX]

    vae = CVAE(img_input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
               n_classes=len(np.unique(y_train)),
               latent_dim=6)

    in1 = np.zeros(shape=(len(one_hot_y_train), x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    in2 = one_hot_y_train
    output = vae([in1, in2])

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    history = vae.fit(x=[x_train, one_hot_y_train],
                      y=x_train,
                      epochs=3000,
                      batch_size=1024)
    vae.save(MODEL_PATH)

    '''
    PLOT
    '''
    plt.plot(history.history['kl_loss'])
    plt.ylabel('kl_loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['reconstruction_loss'])
    plt.ylabel('reconstruction_loss')
    plt.xlabel('epoch')
    plt.show()
