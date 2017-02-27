import sys
import json

import numpy as np
from collections import defaultdict
import scipy.misc

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

# import dataset
import net

np.random.seed(1337)

n = 224
batch_size = 128
nb_epoch = 20
nb_phase_two_epoch = 20
nb_classes = 2


data_directory, model_file_prefix = sys.argv[1:]

datagen = ImageDataGenerator(
    rescale = 1/255.,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest')

datagen.flow_from_directory(data_directory)

print "loading original inception model"

model = net.build_model(nb_classes)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# train the model on the new data for a few epochs

print "training the newly added dense layers"


net.save(model, tags, model_file_prefix)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

print "fine-tuning top 2 inception blocks alongside the top dense layers"

for i in range(1,11):
    print "mega-epoch %d/10" % i
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_phase_two_epoch,
            validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
            nb_val_samples=X_test.shape[0],
            )

    evaluate(model, str(i).zfill(3)+".png")

    net.save(model, tags, model_file_prefix)