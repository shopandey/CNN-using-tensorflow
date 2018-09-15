# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 20:41:14 2018

@author: Shopan Dey
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
import os
import cv2

train_images = []
test_images = []
train_labels = []
test_labels = []
labels = []
test_image_name = []

num_classes = 0
batch_size = 10
epochs = 10

TRAINING_IMAGES_DIR = os.getcwd() + "/train/"

TEST_IMAGES_DIR = os.getcwd() + "/test/"
#count = 0

img_rows = 28
img_cols = 28

for l in os.listdir(TRAINING_IMAGES_DIR):
    print(l)
    labels.append(l)
    TRAINING_IMAGES_SUB_DIR = TRAINING_IMAGES_DIR+l
    for train_image in os.listdir(TRAINING_IMAGES_SUB_DIR):
        
        image_size=28
        filename = os.path.join(TRAINING_IMAGES_SUB_DIR,train_image)
        image = cv2.imread(filename) # read image using OpenCV
        print(filename)
        # Resize image to desired size and preprocess exactly as done during training...
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_images.append(image)
        train_labels.append(num_classes)
    num_classes = num_classes+1    

train_images = np.array(train_images, dtype=np.uint8)
#train_images = train_images.astype('float32')
#train_images = np.multiply(train_images, 1.0/255.0)
#print(train_images.shape)
#train_labels = tf.keras.utils.to_categorical(train_labels,2)

for test_image in os.listdir(TEST_IMAGES_DIR):
         
     filename = os.path.join(TEST_IMAGES_DIR,test_image)
 
     image_size=28
     num_channels=10 # 10 digits 0 to 10
     image = cv2.imread(filename) # read image using OpenCV
     
     # Resize image to desired size and preprocess exactly as done during training...
     image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     test_images.append(image)
     test_labels.append(0)
     test_image_name.append(test_image)
     
test_images = np.array(test_images, dtype=np.uint8)
#test_images = test_images.astype('float32')
#test_images = np.multiply(test_images, 1.0/255.0)
#test_labels = tf.keras.utils.to_categorical(test_labels,2)
#print(test_images.shape)
x_train = train_images
x_test = test_images

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = train_labels
y_test = test_labels
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
#print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
#print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

#Convolution2D(number_filter,row_size, column_size, input_shape=(number_channel, img_row, ima_col))


#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28, 1)),
#    keras.layers.Dense(512, activation=tf.nn.relu),
#    keras.layers.Dropout(0.2),
#    keras.layers.Dense(2, activation=tf.nn.softmax)
#])

model = Sequential()

model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))    
 
#model = model.load_weights("model.h5")
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=200)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save_weights('./my_model')

#model.load_weights('my_model')

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict(x_test)

i = 0
for a in test_labels:
    print(test_image_name[i])
    j = 0
    for b in labels:
        print(labels[j] + " " +  "{0:.2f}".format(predictions[i,j]*100) + "% confidence")
        j = j+1
    print("--------------------------")
    i = i+1

#model.save("model.pb")
