import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
import numpy as np
import os
import cv2
from random import shuffle

train_data = 'TrainingImagesGS/'
test_data = 'TestImages/'
input_shape = (960,720, 1)
num_category = 2


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
model.add(Conv2D(32, kernel_size=(16, 16), activation='relu', input_shape=input_shape))   # 32 - 3 X 3 filters with ReLU Activation Function
    #Each filter convolves filter creates bitmap, makes 33, 3x3 size relu is better
model.add(Conv2D(64, (8, 8), activation='relu'))   # 64 - 3 X 3 filters with ReLU Activation Function

model.add(MaxPooling2D(pool_size=(4, 4)))   # Max Pool the bitmaps 4 bit squares at a time
    #Takes important features from bitmap, takes max value from bitmap
model.add(Flatten())                        # Flatten the dimensions
    # turns bitmaps to 1D
model.add(Dense(128, activation='relu'))    # Adding a dense layer at the end
    # Normal dense layer of neurons (1D)
model.add(Dense(num_category, activation='softmax'))   # Softmax activation function to get probability distributions,
# Categorical Crossentropy loss function with Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
# categorical_crossentropy is something with logs
# Training Hyperparameters
batch_size = 100   # Mini batch sizes
num_epoch = 10     # Number of epochs to train for
model_log = model.fit(tr_img_data, tr_lbl_data, batch_size=batch_size, epochs=num_epoch, verbose=1)

'''
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
model.save_weights("model_digit.h5")
print("Saved model to disk")
'''
