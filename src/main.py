import numpy as np
import tensorflow as tf

# ae = tf.keras.models.load_model('/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/MNIST_ModelX_PUVAE')
# ae.summary()
from puvae.clean_process import clean_process
from utils import show_two_images_3D

puvae = tf.keras.models.load_model(
    '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/puvae/MNIST_ModelX_ALLfirst_400epochs_z=4_lr=0.001_weight=1|1|1')
puvae.summary()

'''
LOAD DATA
'''
inputs = np.load('/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/adv/mnist/hpba_attack_advs.npy')
# (x_train, _), (_, _) = keras.datasets.mnist.load_data()
# if np.max(x_train) > 1:
#     x_train = x_train.astype("float32") / 255.
n_class = 10

'''
Clean and predict
'''
inputs = inputs[1:20]
pred_labels, best_reconstructed_imgs, input_imgs = clean_process(puvae, inputs, n_classes=n_class)

for idx in range(len(input_imgs)):
    show_two_images_3D(x_28_28_left=input_imgs[idx], x_28_28_right=best_reconstructed_imgs[idx], left_title='Input',
                       right_title=f'Best reconstruction \n(label {pred_labels[idx]})',
                       display=True)
