import tensorflow as tf
import horovod.tensorflow.keras as hvd

import numpy as np
import os, shutil, random, sys

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras import backend as K

import warnings

warnings.filterwarnings("ignore")

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_lms_enabled(True)

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

IMG_SIZE = 2048
BATCH_SIZE = 1
LRATE = 0.1e-4
JOBID = 'r_' + str(random.randint(1, 1000000))
WUP = 5

if len(sys.argv) == 2:
    JOBID = sys.argv[1]

ROOT = '/mnt/bb/$USERID'

#load Keras model
model = DenseNet169(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

#Adding custom Layers
x = model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# creating the final model
modelFinal = tf.keras.Model(inputs = model.input, outputs = predictions)

# compile the model
modelFinal.compile(loss = "binary_crossentropy",
                   optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(lr=LRATE * hvd.size())),
                   metrics=["accuracy"],
                   experimental_run_tf_function=False)

if hvd.rank() == 0:
    print(modelFinal.summary())

verbose = 1 if hvd.rank() == 0 else 0

cbs = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=WUP, verbose=verbose),
    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP,    end_epoch=WUP+10, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+10, end_epoch=WUP+20, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+20, end_epoch=WUP+30, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+30,                   multiplier=1e-3),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    cbs.append(tf.keras.callbacks.ModelCheckpoint(JOBID+'.h5', monitor='val_accuracy', verbose=verbose, save_best_only=True,
                                                  save_weights_only=True, mode='auto', period=1))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                dtype='float32')

train_generator = train_datagen.flow_from_directory(
    '/mnt/bb/liuy7366/train/',
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "binary")

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                dtype='float32')

val_generator = val_datagen.flow_from_directory(
    '/mnt/bb/liuy7366/val/',
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "binary")

#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, dtype='float32')

#test_generator = test_datagen.flow_from_directory(
#    '/mnt/bb/liuy7366/test/',
#    target_size = (IMG_SIZE, IMG_SIZE),
#    batch_size = BATCH_SIZE,
#    class_mode = "binary")

# Train the model
modelFinal.fit(
    train_generator,
    steps_per_epoch=len(train_generator) // hvd.size(),
    epochs = 40,
    verbose = verbose,
    workers=4,
    validation_data = val_generator,
    validation_steps=3 * len(val_generator) // hvd.size(),
    callbacks = cbs)

# Evaluate the model on the full data set.
#score = hvd.allreduce(modelFinal.evaluate(test_generator, verbose=verbose))
#if verbose:
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])

#if hvd.rank() == 0:
#    score_str = str(score[1])[10:18].replace('.', '_')
#    shutil.move(JOBID+'.h5', JOBID+'_'+score_str+'.h5')
