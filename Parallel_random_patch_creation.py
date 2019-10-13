#extract patches from original images using multiprocessing (16 process)
#find . -name "*.png" -size -220k -delete

from __future__ import print_function

import sys, os, shutil, pickle
import numpy as np
import glob
import collections
import multiprocessing

from skimage.io import imsave, imread
from skimage.transform import rotate, resize
from skimage.util import pad, crop

import random

import warnings
warnings.filterwarnings("ignore")

wndSize = 1024

def original2patch(item):
    
    inPath = os.path.join(item.inPath, item.ID)
        
    files = sorted(glob.glob(os.path.join(inPath, '*.tif')))
    
    for fileName in files:
        basename = os.path.splitext(fileName)[0]
        basename = os.path.basename(basename)
#        print(basename)
        
        img = imread(fileName)
        print(basename, img.shape)
        
        width = img.shape[1]
        height = img.shape[0]
        
        i = 0
        while i < 300:
            rx = random.randint(0, width - wndSize)
            ry = random.randint(0, height - wndSize)
            xw = (rx, width - (rx + wndSize))
            yw = (ry, height - (ry + wndSize))
            zw = (0, 0)
            imgC = crop(img, (yw, xw, zw))
#            print(rx, ry, imgC.shape)
            filename = os.path.join(item.outPath, basename + '_{0:06d}'.format(rx) + '_{0:06d}.png'.format(ry))
            imsave(filename, imgC)
            i += 1
#end original2patch

inputPath = './input'
patchPath = './output'

dirs = sorted(os.listdir(inputPath))

FolderPairs = collections.namedtuple('FolderPairs', ['inPath', 'outPath', 'ID'])

pairs = []

for folder in dirs:
    pairs.append(FolderPairs(inPath=inputPath, outPath=patchPath, ID=folder))
    
pairsTuple = tuple(pairs)

pool = multiprocessing.Pool(processes=16)
pool.map(original2patch, pairsTuple)

print("done")
