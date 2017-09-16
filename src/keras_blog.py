import os

import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Reshape, Conv2D, BatchNormalization, Lambda
from keras import applications
from keras import optimizers
from keras import metrics
from keras.callbacks import ModelCheckpoint

from keras.applications.imagenet_utils import preprocess_input

import keras_sim


# dimensions of our images.
img_width, img_height = 150, 150

DATASET_PASCAL = "pascal"
DATASET_DOGSCATS = "dogscats"

DATASET = DATASET_PASCAL
#DATASET = DATASET_DOGSCATS

if DATASET == DATASET_PASCAL:
    DEST_DIR = "../data/pascal/processed/"
    TRAIN_REFEREE_DIR = DEST_DIR + 'd1'
    VAL_REFEREE_DIR = DEST_DIR + 'd2'
    TRAIN_CORRECTOR_DIR = DEST_DIR + 'd2'
    VAL_CORRECTOR_DIR = DEST_DIR + 'd3'
else:
    DOGS_CATS_DIR = "../data/dogscats/"
    TRAIN_REFEREE_DIR = DOGS_CATS_DIR + 'train'
    VAL_REFEREE_DIR = DOGS_CATS_DIR + 'valid'
    TRAIN_CORRECTOR_DIR = DOGS_CATS_DIR + 'd2'
    VAL_CORRECTOR_DIR = DOGS_CATS_DIR + 'd3'

MODELS_DIR = '../models' + "_" + DATASET + "/"
FEATURES_DIR = '../features' + "_" + DATASET + "/"

TOP_WEIGHTS_PATH = os.path.join(MODELS_DIR, "referee_top_weights.best.hdf5")
REF_WEIGHTS_PATH = os.path.join(MODELS_DIR, "referee_weights.best.hdf5")

CVD_DEUTERANOPE = "deuteranope"
CVD_PROTANOPE = "protanope"
CVD_TRITANOPE = "tritanope"
CVD = CVD_TRITANOPE

CVD_FILTER = None
if CVD == CVD_DEUTERANOPE:
    CVD_FILTER = keras_sim.lms2lmsd
elif CVD == CVD_PROTANOPE:
    CVD_FILTER = keras_sim.lms2lmsp
else:
    CVD_FILTER = keras_sim.lms2lmst

CORR_WEIGHTS_PATH = os.path.join(MODELS_DIR, CVD + "_" + "corr_weights.best.hdf5")
FULL_WEIGHTS_PATH = os.path.join(MODELS_DIR, CVD + "_" + "full_weights.best.hdf5")

CORR_CORRECTOR = "corrector"
CORR_LINEAR = "linear"
CORR_NONE = "unfiltered"

train_features_path = os.path.join(FEATURES_DIR, 'train_features.npy')
train_labels_path = os.path.join(FEATURES_DIR, 'train_labels.npy')
val_features_path = os.path.join(FEATURES_DIR, 'val_features.npy')
val_labels_path = os.path.join(FEATURES_DIR, 'val_labels.npy')

def get_generator(dirPath, batch_size=128, shuffle=False, **kwargs):
    datagen = ImageDataGenerator(rescale=1. / 255,
        **kwargs)
    generator = datagen.flow_from_directory(
        dirPath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle)
    return generator

def generate_directory_features(model, dirPath):
    batch_size = 32
    from tqdm import tqdm

    generator = get_generator(dirPath, batch_size=batch_size)
    train_features, train_labels = None, None
    nb_samples = generator.samples
    for i in tqdm(range(nb_samples // batch_size), desc="Precalculating features"):
        X_train, Y_train = generator.next()
        #X_train = preprocess_input(X_train)
        Y_pred = model.predict(X_train, batch_size = batch_size)
        if train_features is None:
            train_features_shape = (nb_samples, ) + Y_pred.shape[1:]
            train_features = np.zeros(train_features_shape)

            train_labels_shape = (nb_samples, ) + Y_train.shape[1:]
            train_labels = np.zeros(train_labels_shape)
        train_features[i * batch_size : (i + 1) * batch_size] = Y_pred
        train_labels[i * batch_size : (i + 1) * batch_size] = Y_train
    return train_features, train_labels

def extract_features(useCache=True):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
        print("Making dir: {}".format(FEATURES_DIR))

    if not useCache or not (os.path.exists(train_features_path) and
        os.path.exists(train_labels_path)):
        print("Generating training features...")
        train_features, train_labels = generate_directory_features(model, TRAIN_REFEREE_DIR)
        print(train_features.shape)
        print(train_labels.shape)
        np.save(open(train_features_path, 'w'), train_features)
        np.save(open(train_labels_path, 'w'), train_labels)

    if not useCache or not (os.path.exists(val_features_path) and
        os.path.exists(val_labels_path)):
        print("Generating validation features...")
        val_features, val_labels = generate_directory_features(model, VAL_REFEREE_DIR)
        print(val_features.shape)
        print(val_labels.shape)
        np.save(open(val_features_path, 'w'), val_features)
        np.save(open(val_labels_path, 'w'), val_labels)

    print("All features precomputed.")

def get_top_model(load_weights=True):
    top_model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Flatten(input_shape=(4, 4, 512)))
    #top_model.add(Flatten(input_shape=(7, 7, 512)))
    #model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(BatchNormalization())
    # top_model.add(Dropout(0.5))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    if DATASET == DATASET_PASCAL:
        top_model.add(Dense(20, activation='sigmoid'))
    else:
        top_model.add(Dense(2, activation='sigmoid'))

    top_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  #loss='mean_squared_error',
                  #loss='categorical_hinge',
                  #loss='mean_absolute_error',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])

    if load_weights and os.path.exists(TOP_WEIGHTS_PATH):
        top_model.load_weights(TOP_WEIGHTS_PATH)

    return top_model

def train_top_model():
    extract_features()

    #batch_size = 512
    batch_size = 2048
    epochs = 100

    print("Loading precomputed features...")
    train_data = np.load(open(train_features_path))
    train_labels = np.load(open(train_labels_path))

    validation_data = np.load(open(val_features_path))
    validation_labels = np.load(open(val_labels_path))

    top_model = get_top_model(load_weights = True)

    # checkpoint
    filepath = TOP_WEIGHTS_PATH
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print("Training model...")
    top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=callbacks_list)

def evaluate_top_model():
    batch_size = 2048
    print("Loading precomputed features...")
    validation_data = np.load(open(val_features_path))
    validation_labels = np.load(open(val_labels_path))

    top_model = get_top_model(load_weights = True)

    print("Evaluating TOP model...")
    result = top_model.evaluate(validation_data, validation_labels,
              batch_size=batch_size)
    print()
    for i, name in enumerate(top_model.metrics_names):
        print("{}: {}".format(name, result[i]))

def get_referee_model(load_weights=True):
    vgg16 = applications.VGG16(include_top=False, weights='imagenet',
    input_shape=(img_width, img_height, 3))
    #print("VGG16 layers: {}", len(vgg16.layers))
    for layer in vgg16.layers[:-2]:
        layer.trainable = False
    for layer in vgg16.layers:
        pass
        #print("{} trainable: {}".format(layer.name, layer.trainable))

    top_model = get_top_model(load_weights)

    referee_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer=optimizers.SGD(lr=1e-3, momentum=0.9)
    referee_model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  #loss='mean_squared_error',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])

    if load_weights and os.path.exists(REF_WEIGHTS_PATH):
        referee_model.load_weights(REF_WEIGHTS_PATH)

    return referee_model

def train_referee_model():
    batch_size = 128
    epochs = 100

    train_generator = get_generator(TRAIN_REFEREE_DIR, batch_size,
        shuffle=True,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=10,
        horizontal_flip=True
    )
    val_generator = get_generator(VAL_REFEREE_DIR, batch_size)
    referee_model = get_referee_model(load_weights = True)

    # checkpoint
    filepath = REF_WEIGHTS_PATH
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    referee_model.fit_generator(train_generator,
              epochs=epochs,
              validation_data=val_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              validation_steps=val_generator.samples // batch_size,
              callbacks=callbacks_list)

def evaluate_referee_model():
    batch_size = 128
    val_generator = get_generator(VAL_REFEREE_DIR, batch_size)

    referee_model = get_referee_model(load_weights = True)

    print("Evaluating Referee model...")
    result = referee_model.evaluate_generator(val_generator,
        val_generator.samples // batch_size)
    print()
    for i, name in enumerate(referee_model.metrics_names):
        print("{}: {}".format(name, result[i]))

def get_corrector_model(load_weights = True):
    corrector_model = Sequential()

    # corrector_model.add(Lambda((lambda x: K.dot(x, K.constant(value=keras_sim.rgb2lms.transpose())) ),
    #                             input_shape=(img_width, img_height, 3)))
    # corrector_model.add(Conv2D(3, (5, 5), activation='relu', padding='same'))
    # #corrector_model.add(Conv2D(3, (5, 5), activation='relu', padding='same'))
    # #corrector_model.add(Conv2D(3, (5, 5), activation='relu', padding='same'))
    # #corrector_model.add(Reshape((img_width, img_height, 3)))
    # corrector_model.add(Lambda(lambda x: K.dot(x, K.constant(value=keras_sim.lms2rgb.transpose()))))

    #corrector_model.add(Flatten(input_shape=(img_width, img_height, 3)))

    linear_corr = keras_sim.get_linear_corr(CVD_FILTER, input_shape=(img_width, img_height, 3))
    corrector_model.add(linear_corr)

    corrector_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    corrector_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    corrector_model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    # corrector_model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
    #     input_shape=(img_width, img_height, 3)))
    # corrector_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # corrector_model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    corrector_model.add(Lambda(lambda x: K.clip(x, 0.0, 1.0)))
    #corrector_model.add(Reshape((img_width, img_height, 3)))
    corrector_model.compile(optimizer='rmsprop',
                  loss='mse', metrics=['accuracy'])

    if load_weights and os.path.exists(CORR_WEIGHTS_PATH):
        corrector_model.load_weights(CORR_WEIGHTS_PATH)
    return corrector_model

def get_full_model(load_weights = True, corrector=CORR_CORRECTOR):
    referee_model = get_referee_model()
    for layer in referee_model.layers: #[:26]
        layer.trainable = False

    bottom_model = None
    if corrector == CORR_CORRECTOR:
        corrector_model = get_corrector_model()
        corrector_model.add(keras_sim.get_cvd_layer(CVD_FILTER))
        bottom_model = corrector_model

    elif corrector == CORR_LINEAR:
        linear_model = keras_sim.make_corr_network(CVD_FILTER)
        linear_model.add(keras_sim.get_cvd_layer(CVD_FILTER))
        bottom_model = linear_model
    elif corrector == CORR_NONE:
        sim_model = keras_sim.make_network(CVD_FILTER)
        bottom_model = sim_model

    full_model = Model(inputs=bottom_model.input,
                       outputs=referee_model(bottom_model.output))
    full_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', metrics.top_k_categorical_accuracy])

    if load_weights and os.path.exists(FULL_WEIGHTS_PATH) and corrector == CORR_CORRECTOR:
        print("Load full model weights: {}".format(FULL_WEIGHTS_PATH))
        full_model.load_weights(FULL_WEIGHTS_PATH)

    return full_model, bottom_model

def save_corrector_weights_from_full_model():
    full_model, corrector_model = get_full_model(load_weights = True)
    corrector_model.save_weights(CORR_WEIGHTS_PATH)

def train_full_model():
    print("Training Corrector for: {}".format(CVD))
    batch_size = 128
    epochs = 30

    train_generator = get_generator(TRAIN_CORRECTOR_DIR, batch_size,
        shuffle=True,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=10,
        horizontal_flip=True
    )
    val_generator = get_generator(VAL_CORRECTOR_DIR, batch_size)
    referee_model = get_referee_model(load_weights = True)
    # checkpoint
    filepath = FULL_WEIGHTS_PATH
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
        save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    full_model, bottom_model = get_full_model(load_weights = True)
    for i, l in enumerate(full_model.layers):
        l.trainable = False
        print(i, l.trainable)

    full_model.fit_generator(train_generator,
              epochs=epochs,
              validation_data=val_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              validation_steps=val_generator.samples // batch_size,
              callbacks=callbacks_list)

def evaluate_full_model(corrector=CORR_CORRECTOR):
    batch_size = 128
    val_generator = get_generator(VAL_CORRECTOR_DIR, batch_size)

    full_model, bottom_model = get_full_model(load_weights = True, corrector=corrector)

    print("Evaluating FULL model: {}...".format(corrector))
    result = full_model.evaluate_generator(val_generator,
        val_generator.samples // batch_size)
    print()
    for i, name in enumerate(full_model.metrics_names):
        print("{}: {}".format(name, result[i]))

def preview():
    corrector_model = get_corrector_model()
    corrector_model.add(keras_sim.get_cvd_layer(CVD_FILTER))
    keras_sim.preview_network([
        keras_sim.make_network(CVD_FILTER),
        keras_sim.make_corr_network(CVD_FILTER),
        corrector_model
    ])

if __name__ == '__main__':
    print("Dataset: {}".format(DATASET))
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print("Making dir: {}".format(MODELS_DIR))

    #train_top_model()
    #evaluate_top_model()
    #evaluate_referee_model()

    #train_referee_model()
    #evaluate_referee_model()

    #evaluate_full_model(CORR_NONE)
    #evaluate_full_model(CORR_LINEAR)
    #preview()

    train_full_model()
    save_corrector_weights_from_full_model()



    evaluate_full_model(CORR_NONE)
    evaluate_full_model(CORR_LINEAR)
    evaluate_full_model(CORR_CORRECTOR)
    preview()
