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


train_data_directory, test_data_director, model_file_prefix = sys.argv[1:]

train_datagen = ImageDataGenerator(
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

test_datagen = ImageDataGenerator(rescale=1./255)

print "loading original inception model"

model = net.build_model(nb_classes)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# train the model on the new data for a few epochs

print "training the newly added dense layers"


net.save(model, model_file_prefix)

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
train_generator = train_datagen.flow_from_directory(train_data_directory, target_size=(224,224), batch_size=64, shuffle=True)
val_generator = test_datagen.flow_from_directory(test_data_director, target_size=(224, 224), batch_size=32, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch=200, nb_epoch=11, validation_data = val_generator, nb_val_samples=12500)