#can not find an existing example of setting the correct parameters
#for skimage.imsave saving bigtiff files. So here it goes:

kwargs = {'bigtiff': True}
imsave(os.path.join(outPath, token+'.tif'), r, check_contrast=False, **kwargs)
