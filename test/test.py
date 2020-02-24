import numpy as np
import rarfile
import matplotlib
import cv2
import tensorflow
from tensorflow import keras
import keras
from keras.datasets import mnist

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    print(X_train.shape[0])