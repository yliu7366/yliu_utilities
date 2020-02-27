from __future__ import print_function

import os, shutil, random, sys
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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

import horovod.tensorflow.keras as hvd

#import ddl

import warnings

warnings.filterwarnings("ignore")

hvd.init()

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

K.set_session(tensorflow.Session(config=config))

img_rows = 256
img_cols = 256

NUM_CLASSES = 6

NUM_BATCHSIZE = 40
NUM_LRATE = 1e-5*hvd.size()
#NUM_LRATE = 1e-4

NUM_MAX_EPOCHS = 400

smooth = 1.

CHUNK_SIZE = 48
S_FREQ = 6

ROOT = 'dataroot'

class TrainDataGenerator(Sequence):
    def __init__(self, rootPath, batchSize, fold, chunks = CHUNK_SIZE):
        self.rootPath = rootPath
        self.batchSize = batchSize
        self.images = None
        self.masks = None
        self.size = 0
        self.chunks = chunks
        self.fold = fold
        self.epoch = 0
        self.load_data()

    def __len__(self):
        return int(np.floor(self.size/self.batchSize))

    def __getitem__(self, index):
        x = self.images[index*self.batchSize:(index+1)*self.batchSize]
        y = self.masks[index*self.batchSize:(index+1)*self.batchSize]
        return np.array(x), np.array(y)
    
    def load_a_chunk(self):
        index = random.randint(1, self.chunks) - 1
        token = 'train'
        
        imgs = np.load(os.path.join(self.rootPath, token + '_img_' + str(self.fold) + '_' +str(index) + '.npy'))
        masks = np.load(os.path.join(self.rootPath, token + '_mask_' + str(self.fold) + '_' +str(index) + '_binary.npy'))
        return imgs, masks

    def load_data(self):
        imgs1, masks1 = self.load_a_chunk()
        imgs2, masks2 = self.load_a_chunk()
        
        self.images = np.concatenate((imgs1, imgs2))
        self.masks = np.concatenate((masks1, masks2))

        self.images = self.images.astype(np.float32)/255.
        self.size = len(self.images)

    def on_epoch_end(self):
        self.epoch += 1

        if self.epoch % S_FREQ  == 0:
            self.load_data()
#end def class DataGenerator

class ValDataGenerator(Sequence):
    def __init__(self, rootPath, batchSize, fold, chunks = CHUNK_SIZE):
        self.rootPath = rootPath
        self.batchSize = batchSize
        self.images = None
        self.masks = None
        self.size = 0
        self.chunks = chunks
        self.fold = fold
        self.epoch = 0
        self.load_data()

    def __len__(self):
        return int(np.floor(self.size/self.batchSize))

    def __getitem__(self, index):
        x = self.images[index*self.batchSize:(index+1)*self.batchSize]
        y = self.masks[index*self.batchSize:(index+1)*self.batchSize]
        return np.array(x), np.array(y)

    def load_a_chunk(self):
#        index = random.randint(1, self.chunks) - 1
        token = 'val'
        index = hvd.rank()

        imgs = np.load(os.path.join(self.rootPath, token + '_img_' + str(self.fold) + '_' +str(index) + '.npy'))
        masks = np.load(os.path.join(self.rootPath, token + '_mask_' + str(self.fold) + '_' +str(index) + '_binary.npy'))
        return imgs, masks

    def load_data(self):
        imgs1, masks1 = self.load_a_chunk()

        self.images = imgs1
        self.masks = masks1

        self.images = self.images.astype(np.float32)/255.
        self.size = len(self.images)

#    def on_epoch_end(self):
#end def class DataGenerator

def combined_entropy_jaccard_loss(y_true, y_pred):
    return K.cast(categorical_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred), 'float32')

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

        sum_jaccard_index += ((intersection+smooth)/(union+smooth))/K.cast(NUM_CLASSES, 'float32')
    return K.cast(sum_jaccard_index, 'float32')

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

    convLast = BatchNormalization()(conv9)

    conv10 = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(convLast)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=hvd.DistributedOptimizer(Adam(lr=lrate)),
                loss=combined_entropy_jaccard_loss,
                metrics=[jaccard_index])

    return model   

#patch by Bryant Nelson
class FixedMetricAverageCallback(hvd.callbacks.MetricAverageCallback):
    def __init__(self, *args):
        super(FixedMetricAverageCallback, self).__init__(*args)
    def _make_variable(self, metric, value):
        return super(FixedMetricAverageCallback, self)._make_variable(metric, K.variable(value))
 
def train_and_predict(postfix, bsize, eps, lrate, fold, weights=None):
    
    if hvd.rank() == 0:
        print('Running with postfix:', postfix, 'batch_size:', bsize, 'epochs:', eps,
          'lr:', lrate, 'weights:', weights)
        

    tempH5file = postfix + '_' + str(random.randint(1,1000000)) + '.h5'
    
    model = get_unet(lrate)
    model_checkpoint = ModelCheckpoint(tempH5file, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=lrate*0.01, verbose=1 if hvd.rank() == 0 else 0)
    early_stopping_monitor = EarlyStopping(patience=20, min_delta=0.0001)
    
    #load previous weights to continue training
    if weights:
        model.load_weights(weights)

    cbs = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
          FixedMetricAverageCallback(),
#          hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1 if hvd.rank() == 0 else 0),
          reduce_lr]

    if hvd.rank() == 0:
        cbs.append(model_checkpoint)

    cbs.append(early_stopping_monitor)

    cw = np.zeros((img_cols, img_rows, NUM_CLASSES))
    cw[:,:,0] += 1.
    cw[:,:,1] += 12.
    cw[:,:,2] += 32.
    cw[:,:,3] += 288.
    cw[:,:,4] += 53.
    cw[:,:,5] += 33.

    train_history = model.fit_generator(TrainDataGenerator(ROOT, bsize, fold),
                                       epochs=eps,
                                       class_weight=cw,
                                       verbose= 1 if hvd.rank() == 0 else 0,
                                       validation_data=ValDataGenerator(ROOT, bsize, fold),
                                       callbacks=cbs)
   
    if hvd.rank() == 0:
        score = np.max(train_history.history['val_jaccard_index'])
    
        #throw away ridiculously low scores
        if score > 0.25:
            score_str = str(score)[:8].replace('.', '_')
            weightsFile = 'weights_' + postfix + '_' + score_str + '.h5'
            shutil.move(tempH5file, weightsFile)
        else:
            os.remove(tempH5file)
#end of train_and_predict

def runBatch(bm, key, fold):

    scores = np.zeros(bm, dtype=np.float32)
    postfix = key
    batchSize = NUM_BATCHSIZE 
    epochs = NUM_MAX_EPOCHS
    lrate = NUM_LRATE
    for i in range(bm):
        if hvd.rank() == 0:
            print('\nBatch:', i, 'fold:', fold)
        train_and_predict(postfix, batchSize, epochs, lrate, fold)

f = 0

if len(sys.argv) == 2:
    runBatch(1, sys.argv[1] + '_f' + str(f), f)
else:
    runBatch(1, 'unet_all_onehot_20200215_f' + str(f), f)

if hvd.rank() == 0:
    print("done.")
