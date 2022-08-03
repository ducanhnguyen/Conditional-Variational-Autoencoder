import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def show_images_3D(x, title="", path=None, display=False):
    plt.title(title)
    if x.shape[-1] == 1:
        plt.imshow(x, cmap="gray")
    else:
        plt.imshow(x)

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight', dpi=600)

    if display:
        plt.show()

def show_two_images_3D(x_28_28_left, x_28_28_right, left_title="", right_title="", path=None, display=False):
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.title.set_text(left_title)
    if x_28_28_left.shape[-1] == 1:
        plt.imshow(x_28_28_left, cmap="gray")
    else:
        plt.imshow(x_28_28_left)

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text(right_title)

    if x_28_28_right.shape[-1] == 1:
        plt.imshow(x_28_28_right, cmap="gray")
    else:
        plt.imshow(x_28_28_right)

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight', dpi=600)

    if display:
        plt.show()

def merge(x_train, y_train):
    _, h, w, c = x_train.shape

    y_train = tf.keras.utils.to_categorical(y_train) # (60000, 10)
    # print(y_train.shape)

    y_train = tf.expand_dims(y_train, axis=2)
    y_train = tf.expand_dims(y_train, axis=3) # (60000, 10, 1, 1)
    # print(f'y_train.shape = {y_train.shape}')

    y_train = tf.tile(y_train, multiples=[1, 1, w, c]) # (60000, 10, 28, 1)
    # print(f'y_train.shape = {y_train.shape}')

    tmp = np.concatenate((x_train, y_train), axis=1) # (60000, 38, 28, 1)
    # print(tmp.shape)
    return tmp