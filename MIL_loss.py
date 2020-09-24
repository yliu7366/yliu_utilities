def MIL_loss(y_true, y_pred):
  neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  pos = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

  #split the predictions to two parts: negative samples and positive samples
  for i in range(len(y_true)):
    if tf.math.argmax(y_true[i]) == 0:
      neg = neg.write(neg.size(), y_pred[i])
    else:
      pos = pos.write(pos.size(), y_pred[i])

  #select top_k in positive sample predictions
  pos_tensor = pos.stack()
  pos_tensor_t = tf.transpose(pos_tensor)
  values, indices = tf.math.top_k(pos_tensor_t, k=1)

  #keep only top_k positive sample predictions
  pos_top_k = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  for i in range(len(indices[1])):
    pos_top_k = pos_top_k.write(pos_top_k.size(), pos_tensor[indices[1][i]])

  #recreate y_true since some positive sample predictions were removed
  y_true_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  y_true_pos = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

  for i in range(neg.size()):
    y_true_neg = y_true_neg.write(y_true_neg.size(), tf.one_hot(0, 2))
  for i in range(len(pos_top_k)):
    y_true_pos = y_true_pos.write(y_true_pos.size(), tf.one_hot(1, 2))

  #convert to tensor so we can use the default categorical_crossentropy loss function on the modified predictions
  y_true_MIL = tf.concat([y_true_neg.stack(), y_true_pos.stack()], axis=0)
  y_pred_MIL = tf.concat([neg.stack(), pos_top_k.stack()], axis=0)
  return tf.keras.losses.categorical_crossentropy(y_true_MIL, y_pred_MIL)
