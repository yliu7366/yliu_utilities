def MIL_categorical_crossentropy(y_true, y_pred):
  pMax = 0
  
  for yt, yp in zip(y_true, y_pred):
    if yt[1] > yt[0]:
      pMax = K.max(pMax, yp[1])
  
  for yt, yp in zip(y_true, y_pred):
    if yt[1] > yt[0]:
      yp[1] = pMax    
  
  return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
