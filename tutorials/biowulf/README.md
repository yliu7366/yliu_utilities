# Deep Learning on Biowulf

## Table of Contents
[Python Enviroment](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#Python)  
[Tensorflow](https://github.com/yliu7366/yliu_utilities/edit/master/tutorials/biowulf/README.md#Tensorflow)  
[PyTorch](https://github.com/yliu7366/yliu_utilities/edit/master/tutorials/biowulf/README.md#PyTorch)  
[Example](https://github.com/yliu7366/yliu_utilities/edit/master/tutorials/biowulf/README.md#Example)  

## Python Environment
Biowulf has many modules already installed but not all dependencies are included in the stock Biowulf modules. For example, openslide and tensorflow are two different modules on Biowulf. Custom python environments provides greater flexibility to manage dependencies. Follow this tutorial to create custom python environments on Biowulf [Conda on Biowulf](https://hpc.nih.gov/docs/diy_installation/conda.html).

## Tensorflow
The original TensorFlow installation instructions doesn't work well on Biowulf. Either conda install or pip install will have the libdevice not found at ./libdevice.10.bc error.
The updated TensorFlow installation instructions included fixes for NVCC, XLA, and libdevice file location issues. Applying the steps listed in the *Ubuntu 22.04* section will fix the problem. [Install Tensorflow](https://www.tensorflow.org/install/pip).

## PyTorch
The official PyTorch installation instructions work well on Biowulf. [Install PyTorch](https://pytorch.org/get-started/locally/).

## Example
