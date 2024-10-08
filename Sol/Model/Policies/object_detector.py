from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


from keras.src.applications import VGG16

def scratch():

    _input = Input((224, 224, 1))

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)
    pool2 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv6)
    pool3 = MaxPooling2D((2, 2))(conv7)

    conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
    conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv8)
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
    pool4 = MaxPooling2D((2, 2))(conv10)

    conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
    conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv12)
    pool5 = MaxPooling2D((2, 2))(conv13)

    flat = Flatten()(pool5)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(4096, activation="relu")(dense1)
    output = Dense(1000, activation="softmax")(dense2)

    vgg16_model = Model(inputs=_input, outputs=output)

class ImageFeatureExtractor(tf.keras.Model):
    """WIP"""

    def __init__(self, dim=(64, 64, 3)):
        self._model = VGG16(weights='imagenet')

    def forward(self, x):
        # x = preprocess_input(x)
        return self._model(x)

    def preprocess_input(self, x):
        pass



