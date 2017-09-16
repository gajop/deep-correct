import os

import keras
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam


import matplotlib.pyplot as plt

import pandas
import numpy as np

#DATA_FOLDER = "../data/dl/"
#DATA_FOLDER = ""
DATA_FOLDER = "bla/"

img_width = 224
img_height = 224
batch_size = 5

class MyDirectoryIterator(DirectoryIterator):
    pass


train_datagen = ImageDataGenerator(
#        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         DATA_FOLDER + 'train',
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#         DATA_FOLDER + 'val',
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode='binary')


import numpy as np

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

from vgg16 import VGG16
model = VGG16(include_top=True, weights='imagenet')

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

def show_samples():
    train_generator = train_datagen.flow_from_directory(
            DATA_FOLDER + 'train',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    imgs, labels = train_generator.next()
    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]
        print(label)
        print(np.argmax(label))
        # findex = samples[1][i]
        # print(findex)
        # print(img.shape)
        # fname = train_generator.filenames[int(findex)]
        # fpath = os.path.join(DATA_FOLDER, 'train', fname)
        # print(fpath)

        #img = image.load_img(fpath, target_size=(224, 224))
        imgOrig = img
        img = img.copy() * 255.0
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)


        #imgOrig = img
        #img = image.img_to_array(img)
        #img = img * 255.0
        #img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img)

        preds = model.predict(img)
        print('Predicted[Original]:', decode_predictions(preds))

        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(imgOrig)
        a.set_title('Original: ')

        a = fig.add_subplot(1, 2, 2)
        daltonized = daltonize(image.img_to_array(imgOrig) / 255.0) * 255.0
        daltonized = image.array_to_img(daltonized)
        plt.imshow(daltonized)

        daltonized = daltonized.copy()
        daltonized = image.img_to_array(daltonized)
        daltonized = np.expand_dims(daltonized, axis=0)
        daltonized = preprocess_input(daltonized)
        preds = model.predict(daltonized)
        print('Predicted[Daltonized]:', decode_predictions(preds))
        #print(preds[:3])
        a.set_title('Color blindness: ')

        plt.show()
        img, label = train_generator.next()

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
            DATA_FOLDER + 'train',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    # validation_generator = test_datagen.flow_from_directory(
    #         DATA_FOLDER + 'val',
    #         target_size=(img_width, img_height),
    #         batch_size=batch_size,
    #         class_mode='categorical')

    #print(train_generator.num_class)
    #print(train_generator.class_indices)

    train_generator.num_class = 1000
    import json
    classIndex = json.load(open("imagenet_class_index.json"))
    #print(classIndex)
    totalFound = 0
    oldNewMapping = {}
    for index, node in classIndex.iteritems():
        folder = node[0]
        if folder in train_generator.class_indices:
            oldNewMapping[train_generator.class_indices[folder]] = int(index)
            totalFound += 1

    print(train_generator.classes)
    print(oldNewMapping)
    for i in range(len(train_generator.classes)):
        train_generator.classes[i] = oldNewMapping[train_generator.classes[i]]
    print(train_generator.classes, train_generator.classes.shape)

    print("Total found: {}".format(totalFound))
    #train_generator.class_indices = newClassIndices

    #displayImages(train_generator)


    model.compile(optimizer=Adam(lr=0.01),
                loss='categorical_crossentropy', metrics=['accuracy'])

    #results = model.evaluate_generator(train_generator, steps=5)
    #print(results)
    #print(model.metrics_names)

    #model.predict_generator(train_generator, steps=100)
    for i in range(10):
        X_train, Y_train = train_generator.next()
        X_orig = X_train.copy()
        X_train = preprocess_input(X_train)
        #X_train = X_train * 255.0
        #results = model.evaluate(X_train, Y_train, batch_size=batch_size)
        results = model.predict(X_train, batch_size=batch_size)

        correct = decode_predictions(results)
        correct = [p[0][1] for p in correct]
        plots(X_orig, titles=correct)

        #print(results)

def fit_network():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense


    print(train_generator.next())
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_generator.num_class))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)
    model.save_weights('first_try.h5')

#show_samples()
evaluate_network()
