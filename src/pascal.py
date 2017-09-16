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

class MyDirectoryIterator(DirectoryIterator):
    pass


train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

# Transformation matrix for Deuteranope (a form of red/green color deficit)
lms2lmsd = np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
# Transformation matrix for Protanope (another form of red/green color deficit)
lms2lmsp = np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
# Transformation matrix for Tritanope (a blue/yellow deficit - very rare)
lms2lmst = np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
# Colorspace transformation matrices
rgb2lms = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
lms2rgb = np.linalg.inv(rgb2lms)
# Daltonize image correction matrix
err2mod = np.array([[0,0,0],[0.7,1,0],[0.7,0,1]])

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

def displayImages(train_generator):
    imgs, labels = train_generator.next()
    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]

        imgOrig = img
        img = img.copy()
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0) * 255.0
        img = preprocess_input(img)

        preds = model.predict(img)
        print
        print('Predicted[Original]:', decode_predictions(preds)[0][0])
        print('Correct[Original]:', decode_predictions(np.expand_dims(label, axis=0))[0][0])
        print
        plt.imshow(imgOrig)
        plt.show()

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


def evaluate_network():
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
        plots(X_orig)

        #print(results)


#show_samples()
evaluate_network()
