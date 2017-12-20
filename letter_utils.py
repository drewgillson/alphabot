import numpy as np
import cv2
import os
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical

K.set_image_data_format('channels_last')

def letterNet(input_shape: object = (28, 28, 1), n_class: object = 26) -> object:
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

def detect(image_np, expected_letter, correct_letter_count = 0, min_certainty = 0.85):
    import PIL.Image as Image
    horizontal_start = 200 + (correct_letter_count * 75)
    horizontal_end = horizontal_start + 150

    vertical_start = 400
    vertical_end = vertical_start + 150

    image_np = image_np[vertical_start:vertical_end, horizontal_start:horizontal_end]

    image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 4, 4, 7, 21)
    image_np = cv2.Canny(image_np, 90, 100)

    crop_imgPIL = Image.fromarray(np.uint8(image_np))
    crop_imgPIL.thumbnail((28,28))
    image_np = np.asarray(crop_imgPIL)

    cv2.imshow('crop', image_np)

    letter = ''
    mean_over_samples.append(np.mean(image_np))

    if expected_letter == 'W':
        # because the W has the most white pixels of all, random combinations of other letters
        # sometimes get misidentified as a W, so let's make sure we are sure!
        min_certainty = 0.95

    if np.std(mean_over_samples[-10:]) < 0.8 and (mean_over_samples[-1:][0] > 2):
        for_pred = image_np.reshape(1, 28, 28, 1).astype('float32') / 255
        y_pred = nn.predict(for_pred, batch_size=1)
        certainty = np.amax(y_pred, 1)
        if certainty > min_certainty:
            letter = chr(np.argmax(y_pred, 1) + ord('A'))

    return (letter, crop_imgPIL)

def train():
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    model = letterNet()
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
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch)

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, 100, 0.1),
                        steps_per_epoch=int(y_train.shape[0] / 100),
                        epochs=20,
                        validation_data=(x_test, y_test),
                        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights('trained_model.h5')
    print('Trained model saved')

    return model

nn = letterNet()
mean_over_samples = []