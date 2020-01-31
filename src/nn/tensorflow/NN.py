import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense


def NN_11(input_size):
    model = Sequential()
    model.add(Conv2D(input_shape=(input_size, input_size, 2), data_format='channels_last', filters=128 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=128 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=120 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=111 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=104 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=103 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=90  ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=80  ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=73  ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=71  ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=64  ,kernel_size=3, strides=1, padding='valid',use_bias=True))
    model.add(Dense(3))
    return model

def NN_17(input_size):
    model = Sequential()
    model.add(Conv2D(input_shape=(input_size, input_size, 2), data_format='channels_last', filters=256 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=256 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=251 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=250 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=240 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=240 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=235 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=233 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=233 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=229 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=225 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=223 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=215 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=214 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=205 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=204 ,kernel_size=3, strides=1, padding='same', use_bias=True))
    model.add(Conv2D(filters=200 ,kernel_size=3, strides=1, padding='valid',use_bias=True))
    model.add(Dense(3))
    return model

