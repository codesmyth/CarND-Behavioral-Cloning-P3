import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import random


data_dir = '/Users/dev/Desktop/new_data/'
input_shape = (160, 320, 3)
# Augement a subset of the driving data by including side cameras and flipping the images
augment_proportion = 0.9


lines = []
with open(data_dir + 'driving_log.csv', mode='rt') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

    # Only Take a fraction
    lines = shuffle(lines)[0:int(len(lines)*0.75)]

train_data, validation_data = train_test_split(lines, test_size=0.2)
number_of_samples = len(train_data)

def simple():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))
    model.summary()
    return model


def nvidia_e2e():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()
    return model


def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model


def retrieve_image(image_path):
    # Take the last entry on the image path from the driving log
    tokens = image_path.split('/')
    filename = tokens[-1]
    # pick up the image from the data directory
    image = cv2.imread(data_dir + 'IMG/' + filename)
    # Drive.py uses RGB, openCV uses BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def flip_image(image, measurement):
    flipped_image = cv2.flip(image, 1)  # 1 => Vertical access
    flipped_measurement = float(measurement) * -1.0
    return flipped_image, flipped_measurement


def generator(samples, batch_size=20, is_training=True):
    num_samples = len(samples)
    measurement_correction = 0.15

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            side_images = []
            side_measurements = []

            for line in batch_samples:
                # center, left and right images added to the images array

                image = retrieve_image(line[0]) # center image
                images.append(image)
                # corresponding steering measurements are appended
                measurement = float(line[3])
                measurements.append(measurement)

            # if training augment the images by including side cameras and flipping the images
            if is_training:
                if random.random() > (1 - augment_proportion):
                    # select left and right camera angles, to be flipped.
                    left = retrieve_image(line[1])
                    side_images.append(left)
                    side_measurements.append(measurement + measurement_correction)  # left

                    right = retrieve_image(line[2])
                    side_images.append(right)
                    side_measurements.append(measurement - measurement_correction)  # right

                    # the side images will also be flipped
                    images = images + side_images
                    measurements = measurements + side_measurements

                    flipped_images = []
                    flipped_image_measurements = []
                    for image, measurement in zip(images, measurements):
                        flipped_image, flipped_measurement = flip_image(image, measurement)
                        flipped_images.append(flipped_image)
                        flipped_image_measurements.append(flipped_measurement)

                    images = images + flipped_images
                    measurements = measurements + flipped_image_measurements

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


training_generator = generator(train_data, batch_size=20, is_training=True)
validation_generator = generator(validation_data, batch_size=20, is_training=False)
number_of_samples = len(train_data) + len(train_data) * augment_proportion * 5 #(center + left + right) + flipped

print("Training data:", len(train_data))
model = nvidia_e2e()
model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
model.fit_generator(training_generator, samples_per_epoch=number_of_samples, validation_data=validation_generator, 
                    nb_val_samples=len(validation_data), nb_epoch=3)
model.save('model.h5')





