import os
import numpy as np
import torch

from skimage.io import imread, imsave

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def save_anns(anns, bn):
  if len(anns) == 0:
    return
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

  img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))

  for ann in sorted_anns:
    m = ann['segmentation']
    color_mask = np.random.random(3)
    img[m] = color_mask

  fName = os.path.join('./test_yl/data/', bn+'_mask.png')

  imsave(fName, (img*255).astype(np.uint8))

INPUT = './test_yl/data/he_sam_test.png'

image = imread(INPUT)[:, :, :3]

sam_checkpoint = 'pretrained_checkpoint/sam_hq_vit_l.pth'
device = 'cuda'
model_type = 'vit_l'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
  model=sam,
  points_per_side=32,
  pred_iou_thresh=0.8,
  stability_score_thresh=0.9,
  crop_n_layers=1,
  crop_n_points_downscale_factor=2,
  min_mask_region_area=100,
)

masks = mask_generator.generate(image)

save_anns(masks, os.path.splitext(os.path.basename(INPUT))[0] + '_' + model_type + '_hq')

print('done.')
