import os

import keras
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.optimizers import SGD, RMSprop, Adam

from keras.preprocessing import image

from keras.models import Model, Sequential
from keras.layers import Input, Lambda

from keras import backend as K

import matplotlib.pyplot as plt

import pandas
import numpy as np

#DATA_FOLDER = "../data/dl/"
#DATA_FOLDER = ""
DATA_FOLDER = "../data/pascal/processed/"
TRAIN_REFEREE_DIR = DATA_FOLDER + 'd1'
TRAIN_REFEREE_DIR = "../data/previews/"
#TRAIN_REFEREE_DIR = "../data/ishihara/"

img_width = 150
img_height = 150
batch_size = 5


import numpy as np

# Transformation matrix for Deuteranope (a form of red/green color deficit)
lms2lmsd = np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
# Transformation matrix for Protanope (another form of red/green color deficit)
lms2lmsp = np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
# Transformation matrix for Tritanope (a blue/yellow deficit - very rare)
lms2lmst = np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
# Colorspace transformation matrices
rgb2lms = np.array([[17.8824,43.5161,4.11935],
                    [3.45565,27.1554,3.86714],
                    [0.0299566,0.184309,1.46709]])
lms2rgb = np.linalg.inv(rgb2lms)
# Daltonize image correction matrix
err2mod = np.array([[0,0,0],[0.7,1,0],[0.7,0,1]])

def cvd(x, cvd_filter):
    orig = x
    x = K.dot(x, K.constant(value=rgb2lms.transpose()))
    #print(x.shape, cvd_filter.shape)

    # rgb - sim_rgb contains the color information that dichromats
    # cannot see. err2mod rotates this to a part of the spectrum that
    # they can see.
    x = K.dot(x, K.constant(value=cvd_filter.transpose()))
    #x = K.dot(x, K.variable(value=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])))
    x = K.dot(x, K.constant(value=lms2rgb.transpose()))
    x = K.clip(x, 0.0, 1.0)

    #error = K.dot((orig - x), K.variable(value = err2mod.transpose()))
    #x = error + orig
    return x

def get_cvd_layer(cvd_filter, **kwargs):
    return Lambda((lambda x: cvd(x, cvd_filter)), **kwargs)

def make_network(cvd_filter):
    model = Sequential()
    model.add(get_cvd_layer(cvd_filter, input_shape=(img_width, img_height, 3)))
    return model


def linear_corr(x, cvd_filter):
    orig = x
    x = K.dot(x, K.constant(value=rgb2lms.transpose()))
    #print(x.shape, cvd_filter.shape)

    # rgb - sim_rgb contains the color information that dichromats
    # cannot see. err2mod rotates this to a part of the spectrum that
    # they can see.
    x = K.dot(x, K.constant(value=cvd_filter.transpose()))
    #x = K.dot(x, K.variable(value=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])))
    x = K.dot(x, K.constant(value=lms2rgb.transpose()))
    x = K.clip(x, 0.0, 1.0)

    error = K.dot((orig - x), K.constant(value = err2mod.transpose()))
    x = error + orig

    x = K.clip(x, 0.0, 1.0)
    return x

def get_linear_corr(cvd_filter, **kwargs):
    return Lambda((lambda x: linear_corr(x, cvd_filter)), **kwargs)

def linear_corr_var(x, cvd_filter):
    orig = x
    x = K.dot(x, K.variable(value=rgb2lms.transpose()))
    #print(x.shape, cvd_filter.shape)

    # rgb - sim_rgb contains the color information that dichromats
    # cannot see. err2mod rotates this to a part of the spectrum that
    # they can see.
    x = K.dot(x, K.variable(value=cvd_filter.transpose()))
    #x = K.dot(x, K.variable(value=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])))
    x = K.dot(x, K.variable(value=lms2rgb.transpose()))
    x = K.clip(x, 0.0, 1.0)

    error = K.dot((orig - x), K.variable(value = err2mod.transpose()))
    x = error + orig

    x = K.clip(x, 0.0, 1.0)
    return x

def get_linear_corr_var(cvd_filter, **kwargs):
    return Lambda((lambda x: linear_corr_var(x, cvd_filter)), **kwargs)

def make_corr_network(cvd_filter):
    model = Sequential()
    model.add(get_linear_corr(cvd_filter, input_shape=(img_width, img_height, 3)))
    return model


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

def get_generator(dirPath, batch_size=8):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        dirPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return generator

def preview_network(models):
    display_generator = get_generator(TRAIN_REFEREE_DIR, batch_size=4)

    if not isinstance(models, list):
        print("NOT")
        models = [models]
    print("Total models: {}".format(len(models)))
    for i in range(1):
        X_train, Y_train = display_generator.next()
        X_orig = X_train.copy()

        row = [X_orig * 255.0]
        for model in models:
            preds = model.predict(X_train)
            row.append(preds * 255.0)
        #X_train = X_train * 255.0
        #results = model.evaluate(X_train, Y_train, batch_size=batch_size)
        #results = model.predict(X_train, batch_size=batch_size)

    #    correct = decode_predictions(results)
    #    correct = [p[0][1] for p in correct]
        plots(np.concatenate(row), rows=len(row))

def preview_all():
    display_generator = get_generator(TRAIN_REFEREE_DIR, batch_size=4)
    X_train, Y_train = display_generator.next()
    X_orig = X_train.copy()

    sims = [X_orig * 255]

    for cvd_filter in [lms2lmsd, lms2lmsp, lms2lmst]:
        model = make_network(cvd_filter)

        preds = model.predict(X_train)
        sims.append(preds * 255.0)
    sims = np.concatenate(sims)
    plots(sims, rows=4)

if __name__ == '__main__':
    #model = make_network(lms2lmst)
    #preview_network(model)
    preview_all()
