"""
  Utility code to extract images from SVS files and save to bigtiff files
"""

import openslide
from skimage.io import imsave
import glob
import os
import sys
import numpy as np

if len(sys.argv) != 3:
  print('Need INPUT_PATH and OUT_PATH. Existing...')
else:
  IN = sys.argv[1]
  OUT = sys.argv[2]
  print('Processing svs files in', IN, 'output path:', OUT)

wsis = glob.glob(os.path.join(IN, '*.svs'))

print('Found', len(wsis), 'svs files')

kwargs = {'bigtiff': True}

for wsi in wsis:
  bn = os.path.splitext(os.path.basename(wsi))[0]
  print('  ', bn)

  img = openslide.OpenSlide(wsi)
  imgNpy = np.array(img.read_region((0,0), 0, (img.dimensions[0], img.dimensions[1])))[:,:,:3]

  imsave(os.path.join(OUT, bn+'.tif'), imgNpy, check_contrast=False, **kwargs)

print('Done.')
