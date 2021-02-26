#finding binary contour using SimpleITK
import os, shutil
import SimpleITK as sitk

import numpy as np

img = sitk.ReadImage('seg0000.tif', sitk.sitkUInt32)
imgNpy = sitk.GetArrayFromImage(img)

imgNpy[imgNpy > 0] = 255
imgNpy = imgNpy.astype(np.uint8)

img = sitk.GetImageFromArray(imgNpy)

contour255 = sitk.BinaryContour(img, fullyConnected=True, backgroundValue=0, foregroundValue=255)

sitk.WriteImage(contour255, 'seg0000.png')

print('done.')
