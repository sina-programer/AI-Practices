from matplotlib import pyplot as plt

import tensorflow as tf
import pandas as pd
import numpy as np
import os


def show_image(image, title='', cmap=plt.cm.gray_r):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()


def predict(image, label):
    pred_ohe = model.predict(np.expand_dims(image, axis=0))
    print(f"Real-Label: {label}   Predicted: {np.argmax(pred_ohe)} ")
    show_image(image, title=label)


X = np.load('Data/x_test.npy') / 255
X = np.expand_dims(X, axis=3)

y = np.load('Data/y_test.npy')
y_ohe = tf.keras.utils.to_categorical(y)  # One Hot Encoding y

model = tf.keras.models.load_model('model.h5')

i = 3
predict(X[i], y[i])
