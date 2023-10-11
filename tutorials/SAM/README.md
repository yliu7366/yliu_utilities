# SAM Experiments

## Table of Contents
* [Results](https://github.com/yliu7366/yliu_utilities/tree/master/tutorials/SAM#results)
* [TODO](https://github.com/yliu7366/yliu_utilities/tree/master/tutorials/SAM#todo)
* [SAM](https://github.com/yliu7366/yliu_utilities/tree/master/tutorials/SAM#sam)
* [SAM-HQ](https://github.com/yliu7366/yliu_utilities/tree/master/tutorials/SAM#sam-hq)

  
## Results
Raw image|SAM|SAM-HQ
---|---|---
<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/data/he_sam_test.jpg" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam/he_sam_test_mask.png" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam-hq/he_sam_test_vit_l_hq_mask.png" /> 
<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/data/sam_test.jpg" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam/sam_test_mask.png" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam-hq/sam_test_vit_l_hq_mask.png" /> 
<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/data/sam_test_1.jpg" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam/sam_test_1_mask.png" />|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/results/sam-hq/sam_test_1_vit_l_hq_mask.png" /> 

## TODO
* Extend the SAM and SAM-HQ test code to beyond the max input image size of 1024x1024.

## SAM
[SAM Github repository](https://github.com/facebookresearch/segment-anything)  

## SAM-HQ
[SAM-HQ Github repository](https://github.com/SysCV/sam-hq)

Example [code](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/sam-hq_test.py) to run SAM-HQ models on test images.

### Train SAM-HQ locally
SAM-HQ training script requires python 3.8 as torch.distributed.launch is deprecated in newer python versions. Higher version python such as 3.10 will throw out launcher related errors. Follow the SAM-HQ conda environment [example](https://github.com/SysCV/sam-hq#example-conda-environment-setup) to setup a python 3.8 environment and install scikit-image too.
