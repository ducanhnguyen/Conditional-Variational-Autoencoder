import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

from src.GaussianSampling import GaussianSampling


class PuVAE(Model):
    def __init__(self, img_input_shape, n_classes, latent_dim=16, target_classifier=None, **kwargs):
        super(PuVAE, self).__init__(**kwargs)
        height, width, channels = img_input_shape
        self.target_classifier = target_classifier

        '''
        ENCODER
        '''
        encoder_inputs = keras.Input(shape=(height + n_classes, width, channels), name="img+one-hot class")
        x = Conv2D(32, (7, 7), dilation_rate=2, activation='relu', name="dilated_conv1")(encoder_inputs)
        x = Conv2D(32, (7, 7), dilation_rate=2, activation='relu', name="dilated_conv2")(x)
        # x = Conv2D(32, (7, 7), dilation_rate=2, activation='relu', name = "dilated_conv 3")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        z = GaussianSampling()(mean=z_mean,
                               log_variance=z_log_var)

        encoder = keras.Model(inputs=encoder_inputs,
                              outputs=[z_mean, z_log_var, z],
                              name="encoder")
        # encoder.summary()

        '''
        DECODER
        '''
        latent_inputs = keras.Input(shape=(latent_dim + n_classes,))
        x = layers.Dense(512, activation="relu")(latent_inputs)
        x = layers.Dense(18 * 8 * 32, activation="relu")(x)
        x = layers.Reshape((18, 8, 32))(x)
        while True:
            x = layers.Conv2DTranspose(filters=32, kernel_size=(7, 7), activation="relu", strides=2)(x)
            if x.shape[1] > height and x.shape[2] > width:
                break
        x = layers.Conv2D(filters=1, kernel_size=(x.shape[1] - height + 1, x.shape[2] - width + 1), activation="relu")(
            x)

        # x = layers.Conv2DTranspose(filters=32, kernel_size=(7, 7), activation="sigmoid", strides=2)(x)
        decoder = keras.Model(inputs=latent_inputs,
                              outputs=x,
                              name="decoder")
        # decoder.summary()

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
            self.kl_loss_tracker,
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
        y = model.predict(x, steps=1)

        return y

    def train_step(self, data):
        images, one_hot_labels = data[0]
        merged_inputs = mergeTF(images, one_hot_labels)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(merged_inputs)

            res_weight = 0.01
            kl_weight = 0.1
            ce_weight = 10

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

            ce_loss = tf.reduce_mean(
                tf.keras.metrics.categorical_crossentropy(one_hot_labels,
                                                          self.target_classifier(reconstruction))
            )

            total_loss = res_weight * reconstruction_loss + kl_weight * kl_loss + ce_weight * ce_loss

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


def mergeTF(x_train, one_hot_y_train):
    _, h, w, c = x_train.shape  # (60000, 28, 28, 1)

    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=2)
    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=3)  # (60000, 10, 1, 1)

    one_hot_y_train = tf.tile(one_hot_y_train, multiples=[1, 1, w, c])  # (60000, 10, 28, 1)

    tmp = tf.concat((x_train, one_hot_y_train), axis=1)  # (60000, 38, 28, 1)
    return tmp

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    from matplotlib import pyplot as plt
    from tensorflow import keras

    from puvae.PuVAE import PuVAE

    TARGET_CLASSIFIER = keras.models.load_model(
        '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/target classifier/MNIST_ModelX')
    MODEL_PATH = '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/MNIST_ModelX_PUVAE'
    MAX_INDEX = None

    '''
    LOAD DATA
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    one_hot_y_train = tf.keras.utils.to_categorical(y_train)
    if len(x_train.shape) == 3:  # Ex: (batch, height, width)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    if len(x_test.shape) == 3:  # Ex: (batch, height, width)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    if np.max(x_train) > 1:
        x_train = x_train.astype("float32") / 255

    if np.max(x_test) > 1:
        x_test = x_test.astype("float32") / 255

    '''
    TARGET CLASSIFIER
    '''
    # pred = np.argmax(TARGET_CLASSIFIER.predict(x_train), axis=1)
    # print(f'acc on the training set = {np.sum(pred == y_train) / len(y_train)}')
    # pred = np.argmax(TARGET_CLASSIFIER.predict(x_test), axis=1)
    # print(f'acc on the test set = {np.sum(pred == y_test) / len(y_test)}')

    '''
    TRAIN MODEL
    '''
    x_train = x_train[:MAX_INDEX]
    y_train = y_train[:MAX_INDEX]
    one_hot_y_train = one_hot_y_train[:MAX_INDEX]

    vae = PuVAE(img_input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                n_classes=len(np.unique(y_train)),
                latent_dim=32,
                target_classifier=TARGET_CLASSIFIER)

    in1 = np.zeros(shape=(1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    in2 = tf.expand_dims(one_hot_y_train[0], axis=0)
    output = vae([in1, in2])

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    history = vae.fit(x=[x_train, one_hot_y_train],
                      y=x_train,
                      epochs=3,
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

    plt.plot(history.history['ce_loss'])
    plt.ylabel('ce_loss')
    plt.xlabel('epoch')
    plt.show()
