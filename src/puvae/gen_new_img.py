'''
Test the latent space
'''
import numpy as np
import tensorflow as tf
from utils import show_images_3D

TARGET_CLS_IDX = 0
ENCODER_IDX = 1
DECODER_IDX = 2
LATENT_DIM = 4
N_CLASSES = 10
CVAE = '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/puvae/MNIST_ModelX_ALLfirst_400epochs_z=4_lr=0.001_weight=1|1|1'

'''
LOAD MODEL
'''
ae = tf.keras.models.load_model(CVAE)
ae.summary()

encoder = ae.layers[ENCODER_IDX]
input_encoder = encoder.inputs[0]  # Ex: (None, 38, 28, 1), where 38 = height img + #classes
decoder = ae.layers[DECODER_IDX]
input_decoder = decoder.inputs[0]  # Ex: (None, 16) = latent sapce + n_classes

'''
Generate new images
'''
while True:
    latent = np.random.normal(loc=0.0, scale=1.0, size=LATENT_DIM)

    prob = np.zeros(shape=N_CLASSES)
    label = np.random.randint(0, N_CLASSES)
    prob[label] = 1.0

    input = np.concatenate((latent, prob), axis=0)
    input = input.reshape(-1, LATENT_DIM + N_CLASSES)

    out = decoder.predict(input)

    show_images_3D(out[0], title=f'we want to generate label {label}\nResult:Does it look like {label}?', display=True)
