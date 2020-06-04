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
EPOCHS = 20
LRATE = 0.1e-4
JOBID = 'r_' + str(random.randint(1, 1000000))
WUP = 5

if len(sys.argv) == 2:
    JOBID = sys.argv[1]

ROOT = '/mnt/bb/$USERID'

#load Keras model
model = tf.keras.applications.DenseNet169(weights=None, include_top=True, input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=2)

# compile the model
model.compile(loss = "categorical_crossentropy",
                   optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(lr=LRATE * hvd.size())),
                   metrics=["accuracy"],
                   experimental_run_tf_function=False)

if hvd.rank() == 0:
    print(model.summary())

verbose = 1 if hvd.rank() == 0 else 0

cbs = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=WUP, verbose=verbose),
    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP,    end_epoch=WUP+10, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+10, end_epoch=WUP+20, multiplier=0.5),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+20, end_epoch=WUP+30, multiplier=0.25),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=WUP+30,                   multiplier=0.125),
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
    '/mnt/bb/$USERID/train/',
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                dtype='float32')

val_generator = val_datagen.flow_from_directory(
    '/mnt/bb/$USERID/val/',
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")
    
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=2*len(train_generator) // hvd.size(),
    epochs = 20,
    verbose = verbose,
#    workers=4,
    validation_data = val_generator,
    validation_steps=3 * len(val_generator) // hvd.size(),
    callbacks = cbs)
