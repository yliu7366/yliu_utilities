#Use SimpleITK to remove labels less than 4x4=16 pixels in size
#Save out the filtered labels as binary image.

import SimpleITK as sitk

img = sitk.ReadImage('./data/watershed.png')

print(img.GetSize())

labels = sitk.RelabelComponent(sitk.ConnectedComponent(img))
shapeLabelFilter = sitk.LabelShapeStatisticsImageFilter()
shapeLabelFilter.Execute(labels)

print(shapeLabelFilter.GetNumberOfLabels())

relabelMap =  { i : 0 for i in shapeLabelFilter.GetLabels() if shapeLabelFilter.GetNumberOfPixels(i) < 16 }

output = sitk.ChangeLabel(labels, changeMap=relabelMap) 

shapeLabelFilter1 = sitk.LabelShapeStatisticsImageFilter()
shapeLabelFilter1.Execute(output)

print(shapeLabelFilter1.GetNumberOfLabels())

#convert label maps to binary image
labelImg = (output != 0)

sitk.WriteImage(labelImg * 255, './data/filtered.png')

print("done.")
