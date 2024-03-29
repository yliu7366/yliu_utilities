# Grounded-SAM Experiments
## Table of Contents
* [Local Installation](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/README.md#local-installation)
* [Demos](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/README.md#demos)
* [Grounded SAM for Medical Images](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/README.md#grounded-sam-for-medical-images)

## Local Installation
[Grounded-SAM Github repository](https://github.com/IDEA-Research/Grounded-Segment-Anything)

Read this section if you need a little bit more clarification about the local installation steps. The Grounding-DINO module requires a local CUDA installation so a default [pytorch and CUDA installation](https://pytorch.org/get-started/locally/) in conda won't work.  

Download the CUDA library installation file from [Nvidia](https://developer.nvidia.com/cuda-toolkit-archive) and install it to a local folder so it won't mess up system-wide libraries. I choosed to match the download CUDA library version with pytorch CUDA version, which is 11.7 following the official pytorch installation guide as of 09/02/2023.  

Now export the CUDA_HOME environment variable using the CUDA installation folder and follow the Grounded-SAM local installation steps.

By default, Grounded-SAM is expected to be installed in a *'Grounded-Segment-Anything'* folder.  

### Rust and Cargo
The Tag2Text submodule requires rust compiler. Install rust in *Ubuntu 22.04* by running the script below.
```bash
sudo apt install rustc cargo
```
## Demos
### Grounded-SAM-HQ
The command for running the Grounded-SAM-HQ demo doesn't work. 
```bash
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \  # path to sam-hq checkpoint
  --use_sam_hq \  # set to use sam-hq model
  --input_image sam_hq_demo_image.png \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "chair." \
  --device "cuda"
```
Error messages when running the command:
```bash
usage: Grounded-Segment-Anything Demo [-h] --config CONFIG --grounded_checkpoint GROUNDED_CHECKPOINT [--sam_checkpoint SAM_CHECKPOINT]
                                      [--sam_hq_checkpoint SAM_HQ_CHECKPOINT] [--use_sam_hq] --input_image INPUT_IMAGE --text_prompt TEXT_PROMPT
                                      --output_dir OUTPUT_DIR [--box_threshold BOX_THRESHOLD] [--text_threshold TEXT_THRESHOLD] [--device DEVICE]
Grounded-Segment-Anything Demo: error: the following arguments are required: --input_image, --text_prompt, --output_dir/-o
--use_sam_hq: command not found
--input_image: command not found
```
### Grounded-SAM impainting
This demo is particularly interesting but the command running the demo doesn't work.
```bash
CUDA_VISIBLE_DEVICES=0
python grounded_sam_inpainting_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/inpaint_demo.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "bench" \
  --inpaint_prompt "A sofa, high quality, detailed" \
  --device "cuda"
```

Error messages when running the command:
```bash
ImportError: cannot import name 'CLIPTextModelWithProjection' from 'transformers'
```
## Grounded-SAM Tests
### Automatic Labeling 2023-09-04
|Input Image|Mask|RAM|
---|---|---
<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/data/he_sam_test.jpg" width="200px" height="200px">|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/results/automatic_label_mask_he_sam_test.jpg" width="200px" height="200px">|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/results/automatic_label_output_he_sam_test.jpg" width="200px" height="200px">  

Again, fine tunning on biological images are required for the pipeline to work.

### Detect Everything with Text Prompt 2023-09-03
Using the default *grounded_sam_simple_demo.py* with new input files.  
|Prompt/Class|Input Image|GroundingDINO|Grounded-SAM|
---|---|---|---
blood vessel|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/SAM/data/he_sam_test.jpg" width="200px" height="200px">|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/results/groundingdino_annotated_image_he_sam_test.jpg" width="200px" height="200px">|<img src="https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/Grounded-SAM/results/grounded_sam_annotated_image_he_sam_test.jpg" width="200px" height="200px">

Grounded-SAM: input image -> GroundingDINO annotation -> Grounded-SAM annotation

## Grounded-SAM for Medical Images
Replaced the generic large model with BioViL, a BERT-based biomedical relevant vision-language model. [Github repository](https://github.com/rajpurkarlab/Grounded-SAM)

### References
* BioViL [manuscript](https://arxiv.org/pdf/2204.09817.pdf)
* [Understanding BERT architecture](https://medium.com/analytics-vidhya/understanding-bert-architecture-3f35a264b187)
* [GPT vs. BERT](https://www.makeuseof.com/gpt-vs-bert/)
* [Create a vocabulary for NLP tasks in Python](https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html)
