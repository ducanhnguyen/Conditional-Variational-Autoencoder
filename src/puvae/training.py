'''
Training code of PuVAE
'''
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from puvae.PuVAE import PuVAE

EPOCH = 2000
MAX_INDEX = 100
LATENT_DIM = 16
LR = 0.001
RES_WEIGHT = 1
KL_WEIGHT = 1
CE_WEIGHT = 1

TARGET_CLASSIFIER = keras.models.load_model(
    '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/target classifier/CIFAR10_ModelA')
MODEL_NAME = "CIFAR10_ModelA"

if MAX_INDEX is None:
    MODEL_NAME = f'{MODEL_NAME}_ALLfirst_{EPOCH}epochs_z={LATENT_DIM}_lr={LR}_weight={RES_WEIGHT}|{KL_WEIGHT}|{CE_WEIGHT}'
else:
    MODEL_NAME = f'{MODEL_NAME}_{MAX_INDEX}first_{EPOCH}epochs_z={LATENT_DIM}_lr={LR}_weight={RES_WEIGHT}|{KL_WEIGHT}|{CE_WEIGHT}'
MODEL_PATH = f'/Users/ducanhnguyen/Documents/testingforAI-vnuuet/c-vae/data/autoencoder/puvae/{MODEL_NAME}'

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
