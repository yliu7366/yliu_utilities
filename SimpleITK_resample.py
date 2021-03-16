#Image resampling in SimpleITK
import SimpleITK as sitk
import numpy as np

img = sitk.ReadImage('m1057-1.nrrd')

factor = 10

outputSize = img.GetSize()
outputSize = [outputSize[0], outputSize[1], outputSize[2] * factor]

outputSpacing = img.GetSpacing()
outputSpacing = [outputSpacing[0], outputSpacing[1], outputSpacing[2] / factor]

newImg = sitk.Resample(img, outputSize, sitk.Transform(), sitk.sitkNearestNeighbor, img.GetOrigin(), outputSpacing, img.GetDirection(), 0, img.GetPixelID())
sitk.WriteImage(newImg, '../m1057-1-resampled.nrrd')
