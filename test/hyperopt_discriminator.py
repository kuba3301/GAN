import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Input, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def build_discriminator():
    input_img = (28, 28, 1)

    model = Sequential()

    model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128,256])}}, kernel_size=3,
                    strides=2, input_shape=input_img, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                    kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(BatchNormalization(momentum=0.8))

    if {{choice(['four', 'five', 'six'])}} == 'four':
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))

    elif {{choice(['four', 'five', 'six'])}} == 'four':
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))

    elif {{choice(['four', 'five', 'six'])}} == 'six':
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D({{choice([4, 8, 16, 32, 64, 128, 256])}},
                        kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    img = Input(shape=input_img)
    validity = model(img)

    return Model(img, validity)
