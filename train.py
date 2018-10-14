from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import numpy as np
from tqdm import tqdm
import os
import cv2
from random import shuffle

train_data = 'TrainingImagesGS/'
test_data = 'TestImages/'

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'TrainingImagesGS/open_hand':
        ohl = np.array([1,0])
    elif label == 'TrainingImagesGS/closed_fist':
        ohl = np.array([0,1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if path != "TrainingImagesGS/.DS_Store":
            train_images.append([np.array(img), one_hot_label(path)])
    shuffle(train_images)
    return train_images
'''
def test():
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path)
        print(path)

test()

'''
training_images = train_data_with_label()
# testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,960,720,1)
tr_lbl_data = np.array([i[1] for i in training_images])

# print(len(tr_img_data))
# print(len(tr_lbl_data))



model = Sequential()

# model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid', input_shape=[960,720,1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(x=tr_img_data,y=tr_lbl_data, epochs=5, batch_size=5)
model.summary()
