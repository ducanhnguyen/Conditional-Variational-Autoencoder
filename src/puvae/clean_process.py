'''
DEFENSE.
Given an input (clean or adv), PuVAE would reconstruct this input from the latent space.
'''
import numpy as np
import tensorflow as tf

from puvae.inference import inference
from utils import show_two_images_3D, compute_l2


def clean_process(puvae,
                  input_imgs,  # Ex: (None, 28, 28, 1), pixel in [0, 1]
                  n_classes
                  ):
    pred_labels = []  # the label after applying puvae
    best_reconstructed_imgs = []  # the best reconstruction of input images

    '''
    STEP 1: Use PUVAE to reconstruct input images with different classes.
    For each input image, we have n_class reconstructions.
    '''

    n_img = len(input_imgs)
    all_reconstructions = []
    for idx in range(0, n_classes):
        one_hot_y_train = np.zeros(shape=(n_img, n_classes))
        one_hot_y_train[:, idx] = 1.

        res = inference(puvae, input_imgs, one_hot_y_train)
        all_reconstructions.append(res)
    all_reconstructions = np.asarray(all_reconstructions)  # (#classes, batch, height, width, channel)

    '''
    STEP 2: For each input imaage, find the best reconstruction
    '''
    for idx in range(0, n_img):
        possible_reconstructions = all_reconstructions[:, idx, :, :, :]  # shape = (#classes, height, width, channel)
        ori = input_imgs[idx]  # shape = (height, width, channel)

        l2s = []
        for res_img in possible_reconstructions:
            l2s.append(compute_l2(res_img, ori))
        closest_img_idx = np.argmin(l2s)
        pred_labels.append(closest_img_idx)

        best_reconstructed_imgs.append(possible_reconstructions[closest_img_idx])

    return np.asarray(pred_labels), np.asarray(best_reconstructed_imgs), input_imgs


if __name__ == '__main__':
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
