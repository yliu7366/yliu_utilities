#convert DICOM files to tiffs
import SimpleITK as sitk
import sys, os, shutil
from glob import glob

dicomRoot = 'Imaging_Session_2'
TIFFRoot = 'data/tiff/'

def DICOMtoTiff(dicom, tiff, prefix):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom)

    reader.SetFileNames(dicom_names)
    image = sitk.Cast(reader.Execute(), sitk.sitkUInt16)
        
    for i in range(image.GetSize()[2]):
        basename = prefix + '_' + "{0:04}.tif".format(i)
        sitk.WriteImage(image[:,:,i], os.path.join(tiff, basename))

folders = os.listdir(dicomRoot)

for folder in folders:
    subFolder = os.path.join(dicomRoot, folder)
    if os.path.isdir(subFolder):
        print('Processing', folder, folder)
        DICOMtoTiff(subFolder, TIFFRoot, folder)
        
print("done")
