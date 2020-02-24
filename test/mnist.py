import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#学習時訓練データ抜粋
batch_size = 128
#ラベルの数 0-9
num_classes = 10
#学習回数
epochs = 20

#訓練データセット
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#データ成形-------------

#2D配列を1Dに
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#nn入力のために0.0-1.0に変換
x_train = x_train.astype("float32")
x_train /= 255
x_test = x_test.astype("float32")
x_test /= 255

#ワンホット表現に変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#データ成形-------------


#モデル構築-------------

#モデル作成
model = Sequential()
#入力層(28*28*1)-中間層1(512) 活性化関数relu
model.add(Dense(512, activation="relu", input_shape=(784,)))
#入力ユニットをドロップする割合 精度向上
model.add(Dropout(0.2))
#中間層1(512)-中間層2(512) 活性化関数relu
model.add(Dense(512, activation="relu"))
#入力ユニットをドロップする割合 精度向上
model.add(Dropout(0.2))
#中間層2(512)-出力層(10) 活性化関数softmax
model.add(Dense(10, activation="softmax"))
#モデルの要約を出力
model.summary()
#訓練課程の設定 損失関数 多クラス交差エントロピー
#              最適化アルゴリズム RMSprop
#              評価関数 accuracy 
model.compile(loss="categorical_crossentropy",
optimizer=RMSprop(),
metrics=["accuracy"])

#モデル構築-------------

#学習-------------

history = model.fit(
x_train, y_train,       #訓練データー ラベル
batch_size=batch_size,  #バッチサイズ
epochs=epochs,          #エポック数
verbose=1,              #ログ出力 on=1 off=0
validation_data=(x_test, y_test)) #過学習しないためのデータセット訓練と検証データとは別のほうが良い

#学習-------------

#保存-------------

noise = check_noise
gen_imgs = self.generator.predict(noise)

# 0-1 rescale
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt, :, :, :])
        axs[i, j].axis('off')
        cnt += 1
fig.savefig('images/gen_imgs/kill_me_%d.png' % iteration)

plt.close()

#保存-------------

#評価-------------

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#評価-------------
