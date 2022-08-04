import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import show_images_3D

ae = tf.keras.models.load_model('/data/autoencoder/cifar10_cvae')
ae.summary()

encoder = ae.layers[0]
input_encoder = encoder.inputs[0]  # (None, 38, 28, 1)
decoder = ae.layers[1]
input_decoder = decoder.inputs[0]  # (None, 16) = latent sapce + n_classes

'''
'''
latent_dim_len = 6
n_class = 10

while True:
    latent = np.random.normal(loc=0.0, scale=1.0, size=latent_dim_len)

    prob = np.zeros(shape=10)
    label = np.random.randint(0, 9)
    prob[label] = 1

    input = np.concatenate((latent, prob), axis=0)
    input = input.reshape(-1, latent_dim_len + n_class)

    out = decoder.predict(input)

    show_images_3D(out[0], title=f'we want to generate number {label}\nResult:Does it look like {label}?', display=True)
