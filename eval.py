from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as kbackend
import os
import cv2
import numpy as np
from random import shuffle

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

test_data_paired = test_data_with_label()

x_test = np.stack(get_x_data(test_data_paired), axis=0)
y_test = np.vstack(get_y_data(test_data_paired))

print("x_test shape:", x_test.shape)   # Shape should be 60000 X 28 X 28
print("y_test shape:", y_test.shape)   # Shape should be 60000 X 1

img_rows, img_cols = 960, 720
num_category = 5

if kbackend.image_data_format() == 'channels_first':  # 1 is used here because MNIST is B&W
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_test = x_test.astype('float32')
x_test /= 255

y_test = keras.utils.to_categorical(y_test, num_category)

json_file = open("model_digit.json", "r")

loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model_digit.h5")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


score = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
