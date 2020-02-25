import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Input, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(0)
np.random.RandomState(0)

indexs = 100

dir_path = "./img/generator_model_read/"

f_model = "./model/no_seed/"

model_filename = 'cnn_model.json'
weights_filename = 'cnn_model_weights.hdf5'

json_string = open(os.path.join(f_model, model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join(f_model, weights_filename))

model.summary()

for index in range(indexs):
    noise = np.random.uniform(-1, 1, (1, 100))

    imgs = model.predict(noise)
    B, H, W, C = imgs.shape
    batch = imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num*H, w_num*W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y*H:(y+1)*H, x*W:(x+1)*W] = batch[i, ..., 0]
    fname = str(index).zfill(len(str(3000))) + '.jpg'
    save_path = os.path.join(dir_path, fname)
    plt.imshow(out, cmap='gray')
    plt.title("index: {}".format(index))
    plt.axis("off")
    plt.savefig(save_path)
