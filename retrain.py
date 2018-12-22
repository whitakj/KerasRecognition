# Keras CNN for MNIST Classification - Written By: Sharan Ramjee and adapted from the Keras GitHub Repo

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as kbackend
import os
import cv2
import numpy as np
from random import shuffle


train_data = 'TrainingImages/'
test_data = 'TestImages/'

def one_hot_label(img):
    # Labels training image with a one hot label based on file name
    label = img.split('.')[0] # Ignore everything after the first "." in the file name
    if label == 'open_hand':
        ohl = 0
    elif label == 'closed_fist':
        ohl = 1
    elif label == 'one_finger':
        ohl = 2
    elif label == 'two_fingers':
        ohl = 3
    elif label == 'thumbs_up':
        ohl = 4
    return ohl

def train_data_with_label():
    # Pairs training images with a one hot label in a list in a numpy array
    train_images = []
    for i in os.listdir(train_data):
        path = os.path.join(train_data, i)
        img = cv2.imread(path)
        path = i.split(".")[0]
        if path in ['thumbs_up', 'two_fingers','one_finger','closed_fist','open_hand']:
            train_images.append([np.array(img), one_hot_label(path)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    # Pairs testing images with a one hot label in a list in a numpy array
    test_images = []
    for i in os.listdir(test_data):
        path = os.path.join(test_data, i)
        img = cv2.imread(path)
        path = i.split(".")[0]
        if path in ['thumbs_up', 'two_fingers','one_finger','closed_fist','open_hand']:
            test_images.append([np.array(img), one_hot_label(path)])
    shuffle(test_images)
    return test_images

def get_x_data(data):
    x_data = []
    for i in data:
        x_data.append(i[0])
    return x_data

def get_y_data(data):
    y_data = []
    for i in data:
        y_data.append(i[1])
    return y_data

# Loading data from the MNIST Database
train_data_paired = train_data_with_label()
test_data_paired = test_data_with_label()

x_train = np.stack(get_x_data(train_data_paired), axis=0)
y_train = np.vstack(get_y_data(train_data_paired))
x_test = np.stack(get_x_data(test_data_paired), axis=0)
y_test = np.vstack(get_y_data(test_data_paired))

print("x_train shape:", x_train.shape)   # Shape should be 60000 X 28 X 28
print("y_train shape:", y_train.shape)   # Shape should be 60000 X 1
print("x_test shape:", x_test.shape)   # Shape should be 60000 X 28 X 28
print("y_test shape:", y_test.shape)   # Shape should be 60000 X 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE: Change this when working with other images for the project
img_rows, img_cols = 960, 720
num_category = 5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'
if kbackend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)   # 1 is used here because MNIST is B&W
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Convert datatypes of the numpy arrays to 32 bit floats and divide by 255 (batch normalization)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)     # Shape should be 60000 X 28 X 28 X 1
print(x_train.shape[0], 'train samples')   # 60000 Training Samples
print(x_test.shape[0], 'test samples')     # 10000 Testing Samples

# Convert the labels to one hot form
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# Creating the Model
model = Sequential()
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=input_shape))   # 32 - 3 X 3 filters with ReLU Activation Function
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))   # Max Pool the bitmaps 4 bit squares at a time
model.add(Conv2D(128, (4, 4), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(256, (4, 4), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())                        # Flatten the dimensions
model.add(Dropout(.25))
model.add(Dense(128, activation='relu'))    # Adding a dense layer at the end
model.add(Dense(num_category, activation='softmax'))   # Softmax activation function to get probability distributions
# Categorical Crossentropy loss function with Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Training Hyperparameters
batch_size = 1   # Mini batch sizes
num_epoch = 7    # Number of epochs to train for

model_log = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test), callbacks = [cp_callback])

model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
model.save_weights("model_digit.h5")
print("Saved model to disk")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
