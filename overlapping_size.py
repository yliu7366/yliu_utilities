#calculate new image size for overlapping patch extraction and stitching
#use newSize // wndSize for patch extraction
#img_size: input image size, for example [121, 610, 1280]
#padding: paddings to the input image, for example, [16, 16, 16]
#pch_size: patch size, for example [ 64, 128, 128]
#pch_overlapping: patch overlapping, for example, [16, 16, 16]
#wndSize: effective patch size, equals patch size minus patch_overlapping * 2
def overlappingSize(img_size, padding, pch_size, pch_overlapping):
  newSize = []
  
  for i, p, ps, po in zip(img_size, padding, pch_size, pch_overlapping):
    wndSize = ps - po*2
    newSize.append( ((i+p) // wndSize) * wndSize + i )
    
  return newSize
#end def overlappingSize
