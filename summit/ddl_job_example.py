from __future__ import print_function

import os, shutil, random
import numpy as np
import tensorflow

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, SeparableConv2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils import to_categorical

import ddl

import warnings

warnings.filterwarnings("ignore")

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

gpuOptions = tensorflow.GPUOptions(allow_growth=True)
sess = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpuOptions))
K.set_session(sess)

img_rows = 512
img_cols = 512

NUM_CLASSES = 4

print('ddl.size():', ddl.size())

NUM_BATCHSIZE = 16
NUM_LRATE = 1e-4*ddl.size()

NUM_MAX_EPOCHS = 50

smooth = 1.

ROOT = 'datapath'

def load_train_data():
    imgs_train = np.load(os.path.join(ROOT, 'img_20191216_aug_' +str(ddl.rank()) + '.npy'))
    imgs_mask_train = np.load(os.path.join(ROOT, 'labelOnehot_20191216_aug_binary_' +str(ddl.rank()) + '.npy'))

    return imgs_train, imgs_mask_train

def combined_entropy_jaccard_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    return tensorflow.clip_by_value(K.mean(K.categorical_crossentropy(y_true, y_pred)), -1.0e6, 1.0e6)

def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_index(y_true, y_pred)

def jaccard_index(y_true, y_pred, smooth=1):

    sparse_y_true = K.argmax(y_true, axis=-1)
    sparse_y_pred = K.argmax(y_pred, axis=-1)

    sum_jaccard_index = 0.0

    sparse_y_true_flat = K.flatten(sparse_y_true)
    sparse_y_pred_flat = K.flatten(sparse_y_pred)

    for i in range(NUM_CLASSES):
        sparse_y_true_flat_class = K.cast(K.equal(sparse_y_true_flat, i), 'float32')
        sparse_y_pred_flat_class = K.cast(K.equal(sparse_y_pred_flat, i), 'float32')
 
        intersection = K.sum(K.abs(sparse_y_true_flat_class * sparse_y_pred_flat_class))

        union = K.sum(K.abs(sparse_y_true_flat_class)) + K.sum(K.abs(sparse_y_pred_flat_class)) - intersection

        sum_jaccard_index += ((intersection+smooth)/(union+smooth))/float(NUM_CLASSES)
    return sum_jaccard_index

def get_unet(lrate=1e-5):
    
    inputs = Input((img_rows, img_cols, 3))
    
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

    conv10 = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=lrate),
                           loss=combined_entropy_jaccard_loss,
                           metrics=[jaccard_index])

    return model   
 
def train_and_predict(postfix, bsize, eps, lrate, imgs_train, imgs_mask_train, weights=None):
    
    if ddl.rank() == 0:
        print('Running with postfix:', postfix, 'batch_size:', bsize, 'epochs:', eps,
          'lr:', lrate, 'weights:', weights)
        
    tempH5file = postfix + '_' + str(random.randint(1,1000000)) + '.h5'
    
    model = get_unet(lrate)
    model_checkpoint = ModelCheckpoint(tempH5file, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=lrate*0.0001)
    early_stopping_monitor = EarlyStopping(patience=5, min_delta=0.0001)
    
    #load previous weights to continue training
    if weights:
        model.load_weights(weights)
    
    if ddl.rank() == 0:
        train_history = model.fit(imgs_train, imgs_mask_train,
                            batch_size=bsize, epochs=eps, 
                            verbose=1, 
                            shuffle=True,
                            validation_split = 0.1,
                            callbacks=[ddl.DDLCallback, model_checkpoint, reduce_lr, early_stopping_monitor, ddl.DDLGlobalVariablesCallback()])
    else:
        train_history = model.fit(imgs_train, imgs_mask_train,
                            batch_size=bsize, epochs=eps, 
                            verbose=0, 
                            shuffle=True,
                            validation_split = 0.1,
                            callbacks=[ddl.DDLCallback, reduce_lr, early_stopping_monitor, ddl.DDLGlobalVariablesCallback()])
   
    if ddl.rank() == 0:
        score = np.max(train_history.history['val_jaccard_index'])
    
        #throw away ridiculously low scores
        if score > 0.25:
            score_str = str(score)[:8].replace('.', '_')
            weightsFile = 'weights_' + postfix + '_' + score_str + '.h5'
            shutil.move(tempH5file, weightsFile)
        else:
            os.remove(tempH5file)
#end of train_and_predict

def runBatch(bm, key):

    print('loading data...')

    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype(np.float32) / 255.
    
    print('data loading complete')

    scores = np.zeros(bm, dtype=np.float32)
    postfix = key
    batchSize = NUM_BATCHSIZE 
    epochs = NUM_MAX_EPOCHS
    lrate = NUM_LRATE
    for i in range(bm):
        print('\nBatch:', i)
        train_and_predict(postfix, batchSize, epochs, lrate, imgs_train, imgs_mask_train)
    
runBatch(3, 'postfix')

print("done.")

