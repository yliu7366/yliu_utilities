import os, shutil, csv, time
import collections, multiprocessing

import SimpleITK as sitk
import numpy as np

from albumentations import ElasticTransform, Compose

def augment_multi(aug, images):
    target = {}
    for i, image in enumerate(images[1:]):
        target['image'+str(i)] = 'image'
    return Compose(aug, p=1, additional_targets=target)(image=images[0],
                                                        image0=images[1],
                                                        image1=images[2],
                                                        image2=images[3],
                                                        image3=images[4])

def singleProcess(item):
    img = sitk.ReadImage(os.path.join(originals, item.f + '.tiff'))
    msk = sitk.ReadImage(os.path.join(masks, item.f + '_mask.tiff'), sitk.sitkUInt8)
    
    imgNpy = sitk.GetArrayFromImage(img)
    mskNpy = sitk.GetArrayFromImage(msk)
    
    print(item.f, imgNpy.shape)
    
    labelValues = [102, 153, 204, 255]
    labelValuesNew = [50, 100, 150, 200]
    
    mskList = []
    
    for i in range(len(labelValues)):
        mskList.append(np.copy(mskNpy))
        mskList[i][mskList[i] != labelValues[i]] = 0
        mskList[i][mskList[i] > 0] = 1
        
    af = 60
    if imgNpy.shape[0] < imgNpy.shape[1]:
        af = af * imgNpy.shape[0] / 512
    else:
        af = af * imgNpy.shape[1] / 512

    af = 15 if af < 15 else af
    
    np.random.seed(int(time.time() + multiprocessing.current_process().pid*10000))
    
    imgPair = [imgNpy, mskList[0], mskList[1], mskList[2], mskList[3]]
    
    for a in range(50, 125, 25):
        for s in range(10, 101, 10):        
            transformed = augment_multi([ElasticTransform(alpha=a, alpha_affine=af, sigma=s, p=1)], imgPair)
            sitk.WriteImage(sitk.GetImageFromArray(transformed['image'], isVector=True),
                   os.path.join(output, 'originals', item.f+'_'+str(a)+'_'+str(s)+'.png'))
            
            mskList1 = []
            mskList1.append(transformed['image0'])
            mskList1.append(transformed['image1'])
            mskList1.append(transformed['image2'])
            mskList1.append(transformed['image3'])
            
            mskBinary = mskList1[0] + mskList1[1] + mskList1[2] + mskList1[3]
            
            for i in range(len(labelValues)):
                mskList1[i][mskBinary > 1] = 0
                mskList1[i][mskList1[i] > 0] = labelValuesNew[i]
                
            newMsk = mskList1[0] + mskList1[1] + mskList1[2] + mskList1[3]
            
            sitk.WriteImage(sitk.GetImageFromArray(newMsk),
                   os.path.join(output, 'masks', item.f+'_'+str(a)+'_'+str(s)+'.png'))
#end def singleProcess

fileList = []

csvFile = 'class_counts.csv'
with open(csvFile) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if int(row[1]) > 2:
            fileList.append(row[0])

FolderPairs = collections.namedtuple('FolderPairs', ['f'])

pairs = []

for ff in fileList:
    pairs.append(FolderPairs(f=ff))
    
pairsTuple = tuple(pairs)

pool = multiprocessing.Pool(processes=24)
pool.map(singleProcess, pairsTuple)

print('done.')
