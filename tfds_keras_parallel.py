import os, sys, shutil, random, glob
import numpy as np

import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, SeparableConv2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import PIL
from PIL import Image, ImageEnhance

Image.MAX_IMAGE_PIXELS = None

import warnings
warnings.filterwarnings("ignore")

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

print("keras        {}".format(tf.keras.__version__))
print("tensorflow   {}".format(tf.__version__))
print("number of cpu cores {}".format(len(os.sched_getaffinity(0))))

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

IMG_SIZE = 256
SMOOTH = 1.
NUM_CLASSES = 5
NUM_GPUS = strategy.num_replicas_in_sync
#NUM_GPUS = 1
KEYWORD = '2023_03_17_20x'
ROOT = '../../data/5th_round/training_' + str(IMG_SIZE) + '_' + KEYWORD
LRATE = 1e-4*NUM_GPUS
EPOCHS = 150

BATCH_SIZE = 32*NUM_GPUS

INTENSITY_SCALE_UP = 30

if len(sys.argv) == 3:
  JOBID = sys.argv[1]
  FOLD  = sys.argv[2]
else:
  print('Need JOBID and FOLD, exiting.')
  sys.exit()

class DataGen:
  def __init__(self, imgPath, mskPath, names, aug=False):
    self.imgPath = imgPath
    self.mskPath = mskPath
    self.imgs = names
    self.aug = aug

  def __len__(self):
    return len(self.imgs)

  def augment(self, img, msk):
    # HSV color augmentation
    imgHSV = np.array(img.convert('HSV'), dtype=np.int16)
    imgHSV[:, :] += np.array([random.randint(-10, 10), 0, 0]).astype(np.int16)
    imgHSV[imgHSV[:, :, 0] < 0] += np.array([256, 0, 0]).astype(np.int16)
    imgHSV[imgHSV[:, :, 0] > 255] -= np.array([256, 0, 0]).astype(np.int16)

    newImg = Image.fromarray(imgHSV.astype(np.uint8), 'HSV')
    newImg = newImg.convert('RGB')

    # brightness augmentation
    enhancer = ImageEnhance.Brightness(newImg)
    e = random.uniform(0.5, 1.5)
    newImg = enhancer.enhance(e)

    # contrast augmentation
    enhancer = ImageEnhance.Contrast(newImg)
    e = random.uniform(0.5, 1.5)
    newImg = enhancer.enhance(e)

    t = random.randint(0, 6)
    if t == 0:
      return newImg.transpose(PIL.Image.FLIP_LEFT_RIGHT), msk.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    elif t == 1:
      return newImg.transpose(PIL.Image.FLIP_TOP_BOTTOM), msk.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    elif t == 2:
      return newImg.transpose(PIL.Image.ROTATE_90), msk.transpose(PIL.Image.ROTATE_90)
    elif t == 3:
      return newImg.transpose(PIL.Image.ROTATE_180), msk.transpose(PIL.Image.ROTATE_180)
    elif t == 4:
      return newImg.transpose(PIL.Image.ROTATE_270), msk.transpose(PIL.Image.ROTATE_270)
    elif t == 5:
      return newImg.transpose(PIL.Image.TRANSPOSE), msk.transpose(PIL.Image.TRANSPOSE)

    return newImg, msk

  def __classEmbedding__(self, msk):
    msk = msk // INTENSITY_SCALE_UP
    values, counts = np.unique(msk, return_counts=True)

    keep = [0, 1, 3, 5]
    rest = [v for v in values if v not in keep]

    newMsk = np.ndarray(msk.shape, np.uint8)
    newMsk.fill(0)

    for i in range(len(keep)):
      newMsk[ msk==keep[i] ] = i

    for r in rest:
      newMsk[ msk==r ] = len(keep)

    return newMsk

  def __getitem__(self, index):
    n = self.imgs[index]

    i = Image.open(os.path.join(self.imgPath, n))
    m = Image.open(os.path.join(self.mskPath, n))

    if self.aug == True:
      i, m = self.augment(i, m)

    # convert PIL image to numpy array
    i = np.array(i)
    m = self.__classEmbedding__(np.array(m))
    if NUM_CLASSES > 1:
      m = to_categorical(m, NUM_CLASSES)

    i = (i / 128. - 1.).astype(np.float32)
    m = m.astype(np.float32)
    
    return i, m
    
  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i) 
# end class DataGen

def combined_entropy_jaccard_loss(y_true, y_pred):
  return categorical_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)
#end def combined_entropy_jaccard_loss

def categorical_crossentropy(y_true, y_pred):
  return tf.clip_by_value(K.mean(K.categorical_crossentropy(y_true, y_pred)), -1.0e6, 1.0e6)
#end def categorical_crossentropy

def jaccard_loss(y_true, y_pred):
  return 1. - jaccard_index(y_true, y_pred)
#end def jaccard_loss

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
  return sum_jaccard_index
#end def jaccard_index

def f1(y_true_f, y_pred_f):
    smooth = 0.01
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

  base = 32

  conv1 = Conv2D(base, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(base, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(base*2, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(base*2, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(base*4, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(base*4, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(base*8, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(base*8, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(base*16, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(base*16, (3, 3), activation='relu', padding='same')(conv5)

  up6 = concatenate([Conv2DTranspose(base*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(base*8, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(base*8, (3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv2DTranspose(base*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(base*4, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(base*4, (3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(base*2, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(base*2, (3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(base, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(base, (3, 3), activation='relu', padding='same')(conv9)

  convLast = BatchNormalization()(conv9)

  # conv10 = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(convLast)
  conv10 = Conv2D(NUM_CLASSES, (1, 1))(convLast)

  if NUM_CLASSES == 1:
    output = Activation('sigmoid', dtype='float32', name='segmentation')(conv10)
  else:
    output = Activation('softmax', dtype='float32', name='segmentation')(conv10)

  model = Model(inputs=[inputs], outputs=[output])

  if NUM_CLASSES == 1: 
    model.compile(optimizer=Adam(lr=lrate),
                loss=d_loss,
                metrics=[d_coef])
  else:
    model.compile(optimizer=Adam(lr=lrate),
                loss=combined_entropy_jaccard_loss,
                metrics=[jaccard_index])

  model.summary()

  return model
#end def get_unet

with strategy.scope():
  UNet = get_unet(LRATE)

verbose = 1

f5name = JOBID + '_' + str(IMG_SIZE) + '_f' + FOLD
cbs = [ModelCheckpoint(f5name+'.h5', 
                       monitor='val_loss', 
                       verbose=verbose, save_best_only=True,
                       save_weights_only=True, 
                       mode='auto', 
                       period=1)]
cbs.append(ReduceLROnPlateau(monitor='val_loss',
                             factor=0.1,
                             patience=3,
                             min_lr=LRATE*0.001))
cbs.append(EarlyStopping(monitor="val_loss",
                         min_delta=0,
                         patience=5,
                         verbose=verbose,
                         mode="auto",
                         baseline=None,
                         restore_best_weights=False))

nameTrn = np.load(os.path.join(ROOT, 'trn_f'+FOLD+'.npy'))
nameVal = np.load(os.path.join(ROOT, 'val_f'+FOLD+'.npy'))

trn_datagen =   DataGen(os.path.join(ROOT, 'img'),
                        os.path.join(ROOT, 'msk'),
                        nameTrn,
                        aug=True)

val_datagen =   DataGen(os.path.join(ROOT, 'img'),
                        os.path.join(ROOT, 'msk'),
                        nameVal,
                        )

#step_per_epoch = trn_datagen.__len__() // BATCH_SIZE
#val_step_per_epoch = val_datagen.__len__() // (BATCH_SIZE*2)

if NUM_CLASSES == 1:
  trn_dataset = tf.data.Dataset.from_generator(trn_datagen,
                                               output_signature=(
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32)))
  val_dataset = tf.data.Dataset.from_generator(val_datagen,
                                               output_signature=(
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32)))
else:
  trn_dataset = tf.data.Dataset.from_generator(trn_datagen,
                                               output_signature=(
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float32)))
  val_dataset = tf.data.Dataset.from_generator(val_datagen,
                                               output_signature=(
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float32)))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

trn_dataset = trn_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

trn_dataset = trn_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE*2)

# Train the model
hist = UNet.fit(
    trn_dataset,
    validation_data=val_dataset,
    epochs = EPOCHS,
    verbose = verbose,
    workers = len(os.sched_getaffinity(0)),
    callbacks = cbs)
    
print(hist.history)

vloss = 1
vdcoef = 0

if NUM_CLASSES == 1:
  METRIC = 'val_d_coef'
else:
  METRIC = 'val_jaccard_index'

for l, d in zip(hist.history['val_loss'], hist.history[METRIC]):
  if l < vloss:
    vdcoef = d
    vloss = l

print('lowest loss:', vloss, 'highest '+METRIC+':', vdcoef)

f5name_final = f5name + '_' + str(int(vdcoef*1000))

shutil.move(f5name + '.h5', f5name_final+'.h5')

print('done.')
