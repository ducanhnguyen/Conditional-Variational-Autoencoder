import os.path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src.GaussianSampling import GaussianSampling
from utils import compute_l2, merge_img_with_labels, mergeTF


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


if __name__ == '__main__':
    EPOCH = 1000
    MAX_INDEX = 100
    LATENT_DIM = 4
    LR = 0.001
    RES_WEIGHT = 1
    KL_WEIGHT = 1
    CE_WEIGHT = 1

    TARGET_CLASSIFIER = keras.models.load_model(
        '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/target classifier/MNIST_ModelX')
    MODEL_NAME = "MNIST_ModelX"

    if MAX_INDEX is None:
        MODEL_NAME = f'{MODEL_NAME}_ALLfirst_{EPOCH}epochs_z={LATENT_DIM}_lr={LR}_weight={RES_WEIGHT}|{KL_WEIGHT}|{CE_WEIGHT}'
    else:
        MODEL_NAME = f'{MODEL_NAME}_{MAX_INDEX}first_{EPOCH}epochs_z={LATENT_DIM}_lr={LR}_weight={RES_WEIGHT}|{KL_WEIGHT}|{CE_WEIGHT}'
    MODEL_PATH = f'/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/puvae/{MODEL_NAME}'

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
                latent_dim=LATENT_DIM,
                target_classifier=TARGET_CLASSIFIER,
                res_weight=RES_WEIGHT,
                kl_weight=KL_WEIGHT,
                ce_weight=CE_WEIGHT)

    in1 = np.zeros(shape=(1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    in2 = tf.expand_dims(one_hot_y_train[0], axis=0)
    output = vae([in1, in2])

    vae.compile(optimizer=keras.optimizers.Adam())

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=MODEL_PATH,
    #     save_weights_only=False,  # save full model
    #     monitor='loss',
    #     mode='min',
    #     save_best_only=True)

    history = vae.fit(x=[x_train, one_hot_y_train],
                      y=x_train,
                      epochs=EPOCH,
                      batch_size=1024,
                      # callbacks=[model_checkpoint_callback]
                      )
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
