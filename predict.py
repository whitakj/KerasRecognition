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
import time

print("Get ready...")
time.sleep(1)

camera = 0
cap = cv2.VideoCapture(camera)
cap.set(3,960)
cap.set(4,720)

ret, frame = cap.read()

cv2.imshow('frame', frame)
out = cv2.imwrite('PredictData/1.jpg', frame)
cv2.waitKey(1)
time.sleep(1)
cap.release()

# input_path = 'PredictData/1.jpg'
input_data = frame
img_rows, img_cols = 960, 720
num_category = 5

print("shape:", input_data.shape)   # Shape should be 60000 X 28 X 28


# Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'

if kbackend.image_data_format() == 'channels_first':
    input_data = input_data.reshape(1, 3, img_rows, img_cols)   # 1 is used here because MNIST is B&W

else:
    input_data = input_data.reshape(1, img_rows, img_cols, 3)

# Convert datatypes of the numpy arrays to 32 bit floats and divide by 255 (batch normalization)
input = input_data.astype('float32')
input /= 255

json_file = open("model_digit.json", "r")

loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model_digit.h5")
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

prediction = model.predict(input, batch_size=1,verbose=0,steps=None)

print(prediction.shape)
most_certain = max(prediction)
print(most_certain)
decision = np.where(prediction == most_certain)
print(decision)
possibilities = ["Open Hand","Closed Fist","One Finger","Two Fingers","Thumbs Up"]

print(prediction)
print(possibilities[decision])
