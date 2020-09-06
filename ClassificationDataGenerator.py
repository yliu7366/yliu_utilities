class ClassificationDataGenerator(Sequence):
  def __init__(self, inPath, batchSize, aug=False):
    self.inPath = inPath
    self.batchSize = batchSize
    self.aug = aug
    self.inputs, self.targets = self.__loadData()
  
  def __loadData__(self):
    classes = glob.glob(os.path.join(self.inPath, '*'))
    classes = [os.path.basename(f) for f in classes]
    
    dataset = {}
    
    maxx = 0
    for c in classes:
      imgs = glob.glob(os.path.join(self.inPath, c, '*.jpg'))
      random.shuffle(imgs)
      dataset[c] = imgs
      maxx = max(maxx, len(imgs))
    
    #make all classes have equal number of samples
    for c in sorted(dataset):
      #what if the difference is larger than length?
      if len(dataset[c]) < maxx:
        dataset[c] += random.sample(dataset[c], maxx - len(dataset[c]))
        
    inputs = []
    targets = []
    
    for i in range(maxx):
      for c in sorted(dataset):
        inputs.append(dataset[c][i])
      for c in range(len(dataset)):
        targets.append(one_hot(c, len(classes)))
    
    print('Num. of classes', len(classes), 'samples', maxx*len(classes))
    return inputs, targets
    
  def __len__(self):
    return int(np.floor(len(self.inputs)/self.batchSize))

  def __augment__(self, img):
  
    imgHSV = np.array(img.convert('HSV'), dtype=np.int16)
    imgHSV[:,:] += np.array([randint(-45, 45), 0, 0]).astype(np.int16)
    imgHSV[imgHSV < 0] += 256
    imgHSV[imgHSV > 255] -= 256

    newImg = Image.fromarray(imgHSV.astype(np.uint8))
    newImg = newImg.convert('RGB')
        
    t = random.randint(0, 10)
    if t == 0:
      return newImg.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    elif t == 1:
      return newImg.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    elif t == 2:
      return newImg.transpose(PIL.Image.ROTATE_90)
    elif t == 3:
      return newImg.transpose(PIL.Image.ROTATE_180)
    elif t == 4:
      return newImg.transpose(PIL.Image.ROTATE_270)
    elif t == 5:
      return newImg.transpose(PIL.Image.TRANSPOSE)
    elif t == 6:
      enhancer = ImageEnhance.Brightness(newImg)
      return enhancer.enhance(1.5)
    elif t == 7:
      enhancer = ImageEnhance.Brightness(newImg)
      return enhancer.enhance(0.5)
    elif t == 8:
      enhancer = ImageEnhance.Contrast(newImg)
      return enhancer.enhance(1.5)
    elif t == 9:
      enhancer = ImageEnhance.Contrast(newImg)
      return enhancer.enhance(0.5)
      
    return newImg

  def __getitem__(self, index):
    names = self.inputs[index*self.batchSize:(index+1)*self.batchSize]
    lbls = self.targets[index*self.batchSize:(index+1)*self.batchSize]
    
    imgs = []

    for n in names:
      i = Image.open(n)

      if self.aug == True:
        i = self.__augment__(i)

      i = np.array(i, dtype = np.float32)

      imgs.append(i/255.)
      
    return np.array(imgs), np.array(lbls)
#end class ClassificationDataGenerator
