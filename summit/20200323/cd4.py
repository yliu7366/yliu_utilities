import tensorflow as tf
import horovod.tensorflow.keras as hvd

import numpy as np
import os, shutil, random, sys, glob

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, SeparableConv2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

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

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

IMG_SIZE = 512
BATCH_SIZE = 24
LRATE = 0.5e-4
JOBID = 'r_' + str(random.randint(1, 1000000))
WUP = 3

ROOT = '/mnt/bb/USERID/'

if hvd.rank() == 0:
    print('train sample:', len(glob.glob(os.path.join(ROOT, 'trainimg', 'train', '*.png'))))
    print('val sample:', len(glob.glob(os.path.join(ROOT, 'valimg', 'val', '*.png'))))

if len(sys.argv) == 2:
    JOBID = sys.argv[1]

smooth = 0.01

def f1(y_true_f, y_pred_f):
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def d_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return f1(y_true_f, y_pred_f)

def d_loss(y_true, y_pred):
    return -d_coef(y_true, y_pred)

def get_unet(lrate=1e-5):

    inputs = Input((IMG_SIZE, IMG_SIZE, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    convLast = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(convLast)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(loss=d_loss,
                  metrics=[d_coef],
                  optimizer=hvd.DistributedOptimizer(Adam(lr=lrate)),
                  experimental_run_tf_function=False)

    if hvd.rank() == 0:
        print(model.summary)

    return model

UNet = get_unet(LRATE * hvd.size())

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
    cbs.append(ModelCheckpoint(JOBID+'.h5', monitor='val_loss', verbose=verbose, save_best_only=True,
                              save_weights_only=True, mode='auto', period=1))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
train_datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.)

train_img_generator = train_datagen.flow_from_directory(
    os.path.join(ROOT, 'trainimg'),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    batch_size = BATCH_SIZE)

train_msk_generator = train_datagen1.flow_from_directory(
    os.path.join(ROOT, 'trainmsk',),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    color_mode='grayscale',
    batch_size = BATCH_SIZE)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
val_datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.)

val_img_generator = val_datagen.flow_from_directory(
    os.path.join(ROOT, 'valimg'),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    batch_size = BATCH_SIZE)

val_msk_generator = val_datagen1.flow_from_directory(
    os.path.join(ROOT, 'valmsk'),
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    color_mode='grayscale',
    batch_size = BATCH_SIZE)

# Train the model
UNet.fit(
    zip(train_img_generator, train_msk_generator),
    steps_per_epoch = len(train_img_generator) // hvd.size(),
    epochs = 40,
    verbose = verbose,
    workers = 4,
    validation_data = zip(val_img_generator, val_msk_generator),
    validation_steps = 3*len(val_img_generator) // hvd.size(),
    callbacks = cbs)
