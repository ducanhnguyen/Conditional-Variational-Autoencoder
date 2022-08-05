'''
Reconstruct the original image
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import show_two_images_3D, merge_img_with_labels


def inference(puvae,
              x_train,  # shape = (batch, height, width, channel)
              one_hot_y_train  # shape = (batch, #classes)
              ):
    ENCODER_IDX = 1
    DECODER_IDX = 2

    encoder_input = merge_img_with_labels(x_train, one_hot_y_train)
    encoder = puvae.layers[ENCODER_IDX]
    encoder_output = encoder.predict(encoder_input)
    SAMPLING_IDX = 2
    encoder_output = np.asarray(encoder_output[SAMPLING_IDX])

    decoder_input = np.concatenate((encoder_output, one_hot_y_train), axis=1)  # None, 42)
    decoder = puvae.layers[DECODER_IDX]

    out = decoder.predict(decoder_input)
    return out


if __name__ == '__main__':

    ae = tf.keras.models.load_model(
        '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/puvae/CIFAR10_ModelA_100first_2000epochs_z=16_lr=0.001_weight=1|1|1')
    ae.summary()

    '''
    LOAD DATA
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    one_hot_y_train = tf.keras.utils.to_categorical(y_train)
    if len(x_train.shape) == 3:  # Ex: (batch, height, width)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    if len(x_test.shape) == 3:  # Ex: (batch, height, width)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    if np.max(x_train) > 1:
        x_train = x_train.astype("float32") / 255.

    if np.max(x_test) > 1:
        x_test = x_test.astype("float32") / 255.

    '''
    Inference
    '''
    x_train = x_train[1:20]
    one_hot_y_train = one_hot_y_train[1:20]

    out = inference(ae, x_train, one_hot_y_train)
    for idx in range(len(out)):
        show_two_images_3D(x_train[idx], out[idx], left_title='original img', right_title='reconstructed img',
                           display=True)
