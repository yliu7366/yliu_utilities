## 1. Feature Extraction using a Pre-trained Model
### 1.1 Keras/Tensorflow
Visit the [Keras documentation](https://keras.io/api/applications/) to find a suitable model for feature extraction.

* Use the summary() function to list all layers.
```python
inputs = Input(shape=[PATCH_SIZE, PATCH_SIZE, 3])
base_model = tf.keras.applications.ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
base_model.summary()
```

* Build a new model using any immediate layer.
```python
inputs = Input(shape=[PATCH_SIZE, PATCH_SIZE, 3])
base_model = tf.keras.applications.ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_2_relu').output)
model.summary()
```

* Use the new model to predict/extract features.
```python
features = []
batches = [image_data[i:i+BATCH_SIZE] for i in range(0, len(image_data), BATCH_SIZE)]
for b in batches:
  predicts = model.predict_on_batch( np.array(b) )
  for pp in predicts:
    features.append(pp.flatten())
```

## 2. Classification on the extracted features

## 3. Visualization
