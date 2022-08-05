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

def compute_l2(adv: np.ndarray,
               ori: np.ndarray):
    if np.max(adv) > 1:
        adv = adv / 255.
    if np.max(ori) > 1:
        ori = ori / 255.
    adv = np.round(adv, decimals=2)
    ori = np.round(ori, decimals=2)
    return np.linalg.norm(adv.reshape(-1) - ori.reshape(-1))


def compute_l2s(advs: np.ndarray,
                oris: np.ndarray,
                n_features: int):
    if np.max(advs) > 1:
        advs = advs / 255.
    if np.max(oris) > 1:
        oris = oris / 255.
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)
    l2_dist = np.linalg.norm(advs.reshape(-1, n_features) - oris.reshape(-1, n_features), axis=1)
    return l2_dist


def compute_ssim(advs: np.ndarray,
                 oris: np.ndarray):
    '''
    SSIM distance between a set of two images (adv, ori)
    :param advs: (size, width, height, channel). If size = 1, we have one adversarial example.
    :param oris: (size, width, height, channel). If size = 1, we have one original image.
    :return:
    '''
    if np.max(advs) > 1:
        advs = advs / 255.
    if np.max(oris) > 1:
        oris = oris / 255.
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)
    advs = tf.image.convert_image_dtype(advs, tf.float32)
    oris = tf.image.convert_image_dtype(oris, tf.float32)
    ssim = tf.image.ssim(advs, oris, max_val=2.0)
    return ssim.numpy().reshape(-1)

def mergeTF(x_train, one_hot_y_train):
    _, h, w, c = x_train.shape  # (60000, 28, 28, 1)

    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=2)
    one_hot_y_train = tf.expand_dims(one_hot_y_train, axis=3)  # (60000, 10, 1, 1)

    one_hot_y_train = tf.tile(one_hot_y_train, multiples=[1, 1, w, c])  # (60000, 10, 28, 1)

    tmp = tf.concat((x_train, one_hot_y_train), axis=1)  # (60000, 38, 28, 1)
    return tmp

def merge_img_with_labels(x_train, one_hot_y_train):
    _, h, w, c = x_train.shape  # (60000, 28, 28, 1)

    one_hot_y_train = np.expand_dims(one_hot_y_train, axis=2)
    one_hot_y_train = np.expand_dims(one_hot_y_train, axis=3)  # (60000, 10, 1, 1)

    one_hot_y_train = np.tile(one_hot_y_train, reps=[1, 1, w, c])  # (60000, 10, 28, 1)

    tmp = np.concatenate((x_train, one_hot_y_train), axis=1)  # (60000, 38, 28, 1)
    return tmp
