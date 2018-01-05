""" _    _       _           _           _
   / \  | |_ __ | |__   __ _| |__   ___ | |_
  / _ \ | | '_ \| '_ \ / _` | '_ \ / _ \| __|
 / ___ \| | |_) | | | | (_| | |_) | (_) | |_
/_/   \_\_| .__/|_| |_|\__,_|_.__/ \___/ \__|
          |_|
A screen-less interactive spelling primer powered by computer vision

Copyright (C) 2018  Drew Gillson <drew.gillson@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import os
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical

K.set_image_data_format('channels_last')


def letter_net(input_shape: object = (28, 28, 1), n_class: object = 26) -> object:
    model = models.Sequential()

    x = layers.Input(shape=input_shape)

    # First convolutional layer with max pooling
    conv1 = layers.Conv2D(20, (5, 5), padding="same", activation="relu")(x)
    mp1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Second convolutional layer with max pooling
    conv2 = layers.Conv2D(50, (5, 5), padding="same", activation="relu")(mp1)
    mp2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Hidden layer with 500 nodes
    flatten = layers.Flatten()(mp2)
    dense1 = layers.Dense(500, activation="relu")(flatten)

    # Output layer with n_class nodes (one for each possible letter/number we predict)
    output = layers.Dense(n_class, activation="softmax")(dense1)

    model = models.Model([x],[output])

    model.load_weights('trained_model.h5')

    return model


def load_letter_data():
    import PIL.Image as Image
    directory = os.path.dirname(os.path.realpath(__file__))

    x_train, y_train, x_test, y_test = [], [], [], []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    i = 0
    for letter in letters:
        for filename in os.listdir(directory + "/images/" + letter):
            if filename.endswith(".png"):
                i += 1
                arr = np.asarray(Image.open(directory + '/images/' + letter + '/' + filename))
                if (arr.mean() > 2):  # exclude bad input, these images are almost all black
                    letter_as_int = ord(letter) - ord('A')
                    arr = arr.reshape(28, 28, 1)
                    if i % 10 == 0:
                        x_test.append(arr)
                        y_test.append(letter_as_int)
                    else:
                        x_train.append(arr)
                        y_train.append(letter_as_int)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # shuffle arrays but preserve position / index between the X and Y array
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def train():
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    model = letter_net()
    model.summary()

    (x_train, y_train), (x_test, y_test) = load_letter_data()

    log = callbacks.CSVLogger('log.csv')
    tb = callbacks.TensorBoard(log_dir='tensorboard-logs',
                               batch_size=100, histogram_freq=0)
    checkpoint = callbacks.ModelCheckpoint('weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch)

    model.fit_generator(generator=train_generator(x_train, y_train, 100, 0.1),
                        steps_per_epoch=int(y_train.shape[0] / 100),
                        epochs=20,
                        validation_data=(x_test, y_test),
                        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights('trained_model.h5')
    print('Trained model saved')

    return model