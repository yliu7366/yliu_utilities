## 1. Feature Extraction using a Pre-trained Model
### Keras/Tensorflow
Visit the [Keras documentation](https://keras.io/api/applications/) to find a suitable model for feature extraction.

Use the summary() function to list all layers.
```python
inputs = Input(shape=[PATCH_SIZE, PATCH_SIZE, 3])
base_model = tf.keras.applications.ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
base_model.summary()

## Classification on the extracted features

## Visualization
