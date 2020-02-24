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

tf.set_random_seed(0)

class DCGAN():
    def __init__(self):

        self.shape = (28, 28, 1)
        self.z_dim = 100

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_discriminator(self):
        input_img = self.shape

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=input_img, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0 , 1), (0 , 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2 ,padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        img = Input(shape=input_img)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):
        input_img = (self.z_dim,)

        model = Sequential()

        model.add(Dense(128* 7* 7, activation="relu", input_shape=input_img))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=input_img)
        img = model(noise)

        return Model(noise, img)

    def build_combined(self):
        self.discriminator.trainglabel = False
        model = Sequential(self.generator, self.discriminator)

        return model

    def train(self, iterations, batch_size=128, save_interval=50, check_noise=None):

        img_mnist = self.load_img()

        half_batch = int(batch_size / 2)

        for iteration in range(iterations):

            #discriminator--------

            idx = np.random.randint(0, img_mnist.shape[0], half_batch)

            imgs = img_mnist[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #discriminator--------

            #denerator------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))

            g_loss = self.combined.train_on_batch(
                noise, np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            if iteration % save_interval == 0:

                img_dir = "./img/"
                self.save_images(check_noise, iteration, img_dir)

            #denerator------------

    def load_img(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5

        X_train = X_train.reshape(X_train.shape[0],28,28,1)

        return X_train

    def save_images(self, check_noise, index, dir_path):
        noise = check_noise
        imgs = self.generator.predict(noise)
        B, H, W, C = imgs.shape
        batch= imgs * 127.5 + 127.5
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
        plt.title("iteration: {}".format(index))
        plt.axis("off")
        plt.savefig(save_path)

if __name__ == "__main__":
    dcgan = DCGAN()
    check_noise = np.random.uniform(-1, 1, (1,100))
    dcgan.train(iterations=3000, batch_size=128, save_interval=100, check_noise=check_noise)