#randomly select a training set
from __future__ import print_function

import sys, os, shutil, pickle
import numpy as np
import glob

from skimage.io import imsave, imread
from skimage.transform import rotate, resize
from skimage.util import pad, crop

import random

import warnings
warnings.filterwarnings("ignore")

patchPath = 'input'
trainPath = 'output'

files = sorted(glob.glob(os.path.join(patchPath, '*.png')))

print('total number of patches:', len(files))

trainingSize = 200

#random sampling
picked = random.sample(range(len(files)), trainingSize)

print('sampling 200 patches...')
for index in picked:
    basename = os.path.basename(files[index])
    shutil.move(files[index], os.path.join(trainPath, basename))
    
print('done.')
