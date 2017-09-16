import os

import keras
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

import matplotlib.pyplot as plt

import pandas
import numpy as np

from vgg16 import VGG16

#DATA_FOLDER = "../data/dl/"
#DATA_FOLDER = ""
DATA_FOLDER = "../data/pascal/processed/"
TRAIN_REFEREE_DIR = DATA_FOLDER + 'd1'

img_width = 224
img_height = 224
batch_size = 5

def _invoke(f, img):
    new = np.zeros_like(img)
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j, :3] = np.dot(f, img[i, j, :3])
    return new

def daltonize(img, mode='protanope'):
    lms = _invoke(rgb2lms, img)
    if mode == 'protanope':
        error_function = lms2lmsd
    errored = _invoke(lms2lmsd, lms)
    return _invoke(lms2rgb, errored)

model = VGG16(include_top=True, weights='imagenet')

def evaluate_network():
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
            TRAIN_REFEREE_DIR,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    model.compile(optimizer=Adam(lr=0.01),
                loss='categorical_crossentropy', metrics=['accuracy'])

    for i in range(10):
        X_train, Y_train = train_generator.next()
        X_orig = X_train.copy()
        X_train = preprocess_input(X_train)
        #X_train = X_train * 255.0
        #results = model.evaluate(X_train, Y_train, batch_size=batch_size)
        #results = model.predict(X_train, batch_size=batch_size)

    #    correct = decode_predictions(results)
    #    correct = [p[0][1] for p in correct]

        #print(results)


#show_samples()
evaluate_network()
