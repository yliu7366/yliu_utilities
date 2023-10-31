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
### 1.2 PyTorch


## 2. Clustering on the extracted features
Unsupervised clustering using UMAP and HDBSCAN.
```python
features = np.array(features)

standard_embedding = umap.UMAP(random_state=RANDOM_SEED).fit_transform(features)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

labels = hdbscan.HDBSCAN(min_cluster_size=1000).fit_predict(standard_embedding)
uu, cc = np.unique(labels, return_counts=True)
```

## 3. Visualization
Cluster visualization using Matplotlib.
```python
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')

plt.savefig(os.path.join(ROOT, 'clusters', 'clusters.png'))
```
<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/feature-extraction-classification/clusters.png" />

## 4. Create cluster summary images
We could use random samples from each cluster to create cluster summary images.
```python
def createClusterSummary(lbl):
  """
  create a summary image using multiple clustered images
  :param lbl: cluster label
  :return: summary image
  """
  rows = 10
  cols = 20
  fig = np.ndarray((PATCH_SIZE*rows, PATCH_SIZE*cols, 3), dtype=np.uint8)
  fig_mpo = np.ndarray((PATCH_SIZE*rows, PATCH_SIZE*cols, 3), dtype=np.uint8)

  names = glob.glob(os.path.join(ROOT, 'clusters', str(lbl), '*.png'))

  for y in range(rows):
    for x in range(cols):
      index = y*cols + x
      img = imread(names[index])
      fig[y * PATCH_SIZE:(y + 1) * PATCH_SIZE, x * PATCH_SIZE:(x + 1) * PATCH_SIZE] = img[:, :PATCH_SIZE]
      fig_mpo[y * PATCH_SIZE:(y + 1) * PATCH_SIZE, x * PATCH_SIZE:(x + 1) * PATCH_SIZE] = img[:, PATCH_SIZE:]

  return fig, fig_mpo
```
The code block above create cluster summary images using 20x10 images from each cluster.
