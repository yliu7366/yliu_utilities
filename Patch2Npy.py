#convert patches to .npy files for training
from __future__ import print_function

import sys, os, shutil
import numpy as np
import glob
import collections
import multiprocessing

from skimage.io import imsave, imread
from skimage.transform import rotate, resize
from skimage.util import pad, crop

import warnings
warnings.filterwarnings("ignore")

def create_train_data(root, stamp):
    
    print(root)
    
    patches = os.path.join(root, 'patches')
    labelC2 = os.path.join(root, 'labelC2')
    labelC3 = os.path.join(root, 'labelC3')
    
    images = sorted(os.listdir(patches))
    total = len(images)
    
    imgs = np.ndarray((total, 512, 512, 3), dtype=np.uint8)
    imgsC2 = np.ndarray((total, 512, 512), dtype=np.uint8)
    imgsC3 = np.ndarray((total, 512, 512), dtype=np.uint8)

    i = 0
    print('  Loading...')
    for image_name in images:

        img = imread(os.path.join(patches, image_name))
        imgC2 = imread(os.path.join(labelC2, image_name))
        imgC3 = imread(os.path.join(labelC3, image_name))
        
        img = np.array([img])
        imgC2 = np.array([imgC2])
        imgC3 = np.array([imgC3])

        imgs[i] = img
        imgsC2[i] = imgC2
        imgsC3[i] = imgC3

        if i % 100 == 0:
            print('    Done: {0}/{1} images'.format(i, total))
        i += 1
    print('  Loading done.')
    
    np.save(os.path.join(root, 'img_' + stamp + '.npy'), imgs)
    np.save(os.path.join(root, 'labelC2_' + stamp + '.npy'), imgsC2)
    np.save(os.path.join(root, 'labelC3_' + stamp + '.npy'), imgsC3)
    print('  Saving to .npy files done.')
#end create_train_data

root = './input'

create_train_data(root, '20191012')

print("done.")
