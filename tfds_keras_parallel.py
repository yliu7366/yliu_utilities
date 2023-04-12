nameTrn = np.load(os.path.join(ROOT, 'trn_f'+FOLD+'.npy'))
nameVal = np.load(os.path.join(ROOT, 'val_f'+FOLD+'.npy'))

def ImageAugment(img):
  
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

  return newImg
    
def classEmbedding(msk):
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
    
def ValidationDataGen(i):
  i = i.numpy()
  
  img = Image.open(os.path.join(ROOT, 'img', nameVal[i]))
  msk = Image.open(os.path.join(ROOT, 'msk', nameVal[i]))

  # convert PIL image to numpy array
  img = np.array(img)
  msk = classEmbedding(np.array(msk))

  if NUM_CLASSES > 1:
    msk = to_categorical(msk, NUM_CLASSES)

  img = (img / 128. - 1.).astype(np.float32)
  msk = msk.astype(np.float32)
      
  return img, msk
  
def TrainingDataGen(i):
  i = i.numpy()
  
  img = Image.open(os.path.join(ROOT, 'img', nameTrn[i]))
  msk = Image.open(os.path.join(ROOT, 'msk', nameTrn[i]))

  img = ImageAugment(img)
  
  # convert PIL image to numpy array
  img = np.array(img)
  msk = np.array(msk)
  
  t = random.randint(0, 5)
  if t == 0:
    img = np.rot90(img)
    msk = np.rot90(msk)
  elif t == 1:
    img = np.rot90(img, 2)
    msk = np.rot90(msk, 2)
  elif t == 2:
    img = np.rot90(img, 3)
    msk = np.rot90(msk, 3)
  elif t == 3:
    img = np.fliplr(img)
    msk = np.fliplr(msk)
  elif t == 4:
    img = np.flipud(img)
    msk = np.flipud(msk)
  
  msk = ClassEmbedding(msk)

  if NUM_CLASSES > 1:
    msk = to_categorical(msk, NUM_CLASSES)

  img = (img / 128. - 1.).astype(np.float32)
  msk = msk.astype(np.float32)
  
  return img, msk

"""
def FixShape(x, y):
  x.set_shape([None, None, None, channels])
  y.set_shape([None, classes])
  return x, y
"""

def GetTrainingDataset():
  # the index generator
  z = list(range(len(nameTrn)))

  dataset = tf.data.Dataset.from_generator(lambda: z, tf.int64)

  if NUM_CLASSES == 1:
    dataset = dataset.map(lambda i: tf.py_function(func=TrainingDataGen,
                                              inp = [i],
                                              Tout = [tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32)]),
                        num_parallel_calls = tf.data.AUTOTUNE)
  else:
    dataset = dataset.map(lambda i: tf.py_function(func=TrainingDataGen,
                                              inp = [i],
                                              Tout = [tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float32)]),
                        num_parallel_calls = tf.data.AUTOTUNE)
                        
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  
  dataset = dataset.with_options(options)
  dataset = dataset.batch(BATCH_SIZE)
  # dataset = dataset.batch(BATCH_SIZE).map(FixShape)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  
  return dataset
  
def GetValidationDataset():
  # the index generator
  z = list(range(len(nameVal)))

  dataset = tf.data.Dataset.from_generator(lambda: z, tf.int64)

  if NUM_CLASSES == 1:
    dataset = dataset.map(lambda i: tf.py_function(func=TrainingDataGen,
                                              inp = [i],
                                              Tout = [tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE), dtype=tf.float32)]),
                        num_parallel_calls = tf.data.AUTOTUNE)
  else:
    dataset = dataset.map(lambda i: tf.py_function(func=TrainingDataGen,
                                              inp = [i],
                                              Tout = [tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float32)]),
                        num_parallel_calls = tf.data.AUTOTUNE)
                        
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  
  dataset = dataset.with_options(options)
  dataset = dataset.batch(BATCH_SIZE*2)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  
  return dataset
  
trn_dataset = GetTrainingDataset()
val_dataset = GetValidationDataset()
