import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU, \
    Conv1D, Embedding, MaxPooling1D, Flatten

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, x, y):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.initialized = True

    def __len__(self):
        if self.initialized:
            print(f"Using sample size: {len(self.x)} and batch size: {self.batch_size}")
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return x, y


def build_cnn(feature_size, weights):
    model = keras.models.Sequential()
    model.add(Input(shape=(feature_size,)))
    model.add(Embedding(
        input_dim=weights.shape[0], output_dim=weights.shape[1],
        weights=[weights], trainable=False, input_length=feature_size
    ))
    model.add(Conv1D(32, 5, padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPooling1D(padding="same"))
    model.add(Conv1D(64, 5, padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPooling1D(padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    if len(gpus) >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=len(gpus))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def build_lstm_model(feature_size, weights):
    inp = keras.layers.Input(shape=(feature_size,))
    embed = keras.layers.Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                   weights=weights, trainable=False,
                                   input_length=feature_size)(inp)
    spacial_drop = keras.layers.Dropout(0.2)(embed)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(40, return_sequences=True))(spacial_drop)
    gap = keras.layers.AveragePooling1D()(lstm)
    gmp = keras.layers.MaxPooling1D()(lstm)
    concat = keras.layers.concatenate([gap, gmp])
    flat = keras.layers.Flatten()(concat)
    dense = keras.layers.Dense(128, activation="relu")(flat)
    dense = keras.layers.Dense(2, activation="sigmoid")(dense)

    model = keras.models.Model(inputs=inp, outputs=dense)
    if len(gpus) >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=len(gpus))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def build_deep_net(feature_size):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_dim=feature_size,
                                 kernel_initializer="glorot_uniform",
                                 activation="tanh"))
    model.add(keras.layers.Dense(128, input_dim=feature_size,
                                 kernel_initializer="glorot_uniform",
                                 activation="tanh"))
    model.add(keras.layers.UpSampling1D())
    model.add(keras.layers.Dense(16, activation="tanh"))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    if len(gpus) >= 2:
        model = keras.utils.multi_gpu_model(model, gpus=len(gpus))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')
    return model
