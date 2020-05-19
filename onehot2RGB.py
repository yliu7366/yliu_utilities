#example code to convert one hot encoding to RGB colors
def onehot2RGB(onehot, name):
    indexImg = np.argmax(onehot, axis=-1)
    rgbImg = np.ndarray((img_rows, img_cols, 3), dtype=np.uint8)
    rgbImg[indexImg[:,:] == 0] = [0, 0, 0]
    rgbImg[indexImg[:,:] == 1] = [150, 75, 0]
    rgbImg[indexImg[:,:] == 2] = [255, 0, 255]
    rgbImg[indexImg[:,:] == 3] = [0, 255, 0]

    imsave(name, rgbImg)
