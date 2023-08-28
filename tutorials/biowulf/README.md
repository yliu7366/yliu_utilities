# Deep Learning on Biowulf

## Table of Contents
[Custom Python Enviroment](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#custom-python-environment)  
[Tensorflow](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#tensorflow)  
[PyTorch](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#pytorch)  
[Examples](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#examples)  
  * [Python environment for doing vision tasks with TensorFlow](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#python-environment-for-doing-vision-tasks-using-tensorflow)
  * [Job submission on Biowulf](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#job-submission-on-biowulf)
  * [Simple MNIST convnet example](https://github.com/yliu7366/yliu_utilities/blob/master/tutorials/biowulf/README.md#keras-simple-mnist-convent-example)

## Custom Python Environment
Biowulf has many modules already installed but not all dependencies are included in the stock Biowulf modules. For example, openslide and tensorflow are two different modules on Biowulf. Custom python environments provides greater flexibility to manage dependencies. Follow this tutorial to create custom python environments on Biowulf [Conda on Biowulf](https://hpc.nih.gov/docs/diy_installation/conda.html). Biowulf recommends to save conda initalization code into a separate coda init file instead of adding automatic conda initialization into startup files. After conda setup following Biowulf instructions, you will have a conda init file saved in your own Biowulf folder. The conda init file should be sourced in job submission scripts in order to use custom python environment on Biowulf.

## TensorFlow
The original TensorFlow installation instructions doesn't work well on Biowulf. Either conda install or pip install will have the *libdevice not found at ./libdevice.10.bc* error.
The updated TensorFlow installation instructions included fixes for NVCC, XLA, and libdevice file location issues. Applying the steps listed in the *Ubuntu 22.04* section will fix the problem. [Install Tensorflow](https://www.tensorflow.org/install/pip).

Steps to fix the libdevice issue. Copied from the [Tensorflow documentation](https://www.tensorflow.org/install/pip).
```shell
# Install NVCC
conda install -c nvidia cuda-nvcc=11.3.58
# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```

## PyTorch
The official PyTorch installation instructions work well on Biowulf. [Install PyTorch](https://pytorch.org/get-started/locally/).

## Examples

### Python environment for doing vision tasks using TensorFlow
```yaml
name: tensorflow_vision
channels:
  - nvidia
  - conda-forge
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - aom=3.5.0=h27087fc_0
  - asciitree=0.3.3=py_2
  - blosc=1.21.4=h0f2a231_0
  - brotli=1.0.9=h166bdaf_9
  - brotli-bin=1.0.9=h166bdaf_9
  - brotli-python=1.0.9=py39h5a03fae_9
  - brunsli=0.1=h9c3ff4c_0
  - bzip2=1.0.8=h7f98852_4
  - c-ares=1.19.1=hd590300_0
  - c-blosc2=2.10.0=hb4ffafa_0
  - ca-certificates=2023.7.22=hbcca054_0
  - cairo=1.16.0=h35add3b_1015
  - certifi=2023.7.22=pyhd8ed1ab_0
  - cfitsio=4.2.0=hd9d235c_0
  - charls=2.3.4=h9c3ff4c_0
  - charset-normalizer=3.2.0=pyhd8ed1ab_0
  - cuda-nvcc=11.3.58=h2467b9f_0
  - cudatoolkit=11.8.0=h4ba93d1_12
  - dav1d=1.2.1=hd590300_0
  - eigen=3.4.0=h00ab1b0_0
  - entrypoints=0.4=pyhd8ed1ab_0
  - expat=2.5.0=hcb278e6_1
  - fasteners=0.17.3=pyhd8ed1ab_0
  - fftw=3.3.10=nompi_hc118613_108
  - font-ttf-dejavu-sans-mono=2.37=hab24e00_0
  - font-ttf-inconsolata=3.000=h77eed37_0
  - font-ttf-source-code-pro=2.038=h77eed37_0
  - font-ttf-ubuntu=0.83=hab24e00_0
  - fontconfig=2.14.2=h14ed4e7_0
  - fonts-conda-ecosystem=1=0
  - fonts-conda-forge=1=0
  - freetype=2.12.1=hca18f0e_1
  - fsspec=2023.6.0=pyh1a96a4e_0
  - gdk-pixbuf=2.42.8=hff1cb4f_1
  - geojson=3.0.1=pyhd8ed1ab_0
  - geos=3.11.2=hcb278e6_0
  - gettext=0.21.1=h27087fc_0
  - giflib=5.2.1=h0b41bf4_3
  - glib=2.76.4=hfc55251_0
  - glib-tools=2.76.4=hfc55251_0
  - hdf5=1.12.2=nompi_h4df4325_101
  - icu=72.1=hcb278e6_0
  - idna=3.4=pyhd8ed1ab_0
  - imagecodecs=2022.9.26=py39hf32c164_4
  - imageio=2.31.1=pyh24c5eb1_0
  - joblib=1.3.0=pyhd8ed1ab_1
  - jpeg=9e=h0b41bf4_3
  - jxrlib=1.1=h7f98852_2
  - keyutils=1.6.1=h166bdaf_0
  - krb5=1.21.1=h659d440_0
  - lazy_loader=0.2=pyhd8ed1ab_0
  - lcms2=2.14=h6ed2654_0
  - ld_impl_linux-64=2.40=h41732ed_0
  - lerc=4.0.0=h27087fc_0
  - libaec=1.0.6=hcb278e6_1
  - libavif=0.11.1=h8182462_2
  - libblas=3.9.0=17_linux64_openblas
  - libbrotlicommon=1.0.9=h166bdaf_9
  - libbrotlidec=1.0.9=h166bdaf_9
  - libbrotlienc=1.0.9=h166bdaf_9
  - libcblas=3.9.0=17_linux64_openblas
  - libcurl=8.2.1=hca28451_0
  - libdeflate=1.14=h166bdaf_0
  - libedit=3.1.20191231=he28a2e2_2
  - libev=4.33=h516909a_1
  - libexpat=2.5.0=hcb278e6_1
  - libffi=3.4.2=h7f98852_5
  - libgcc-ng=13.1.0=he5830b7_0
  - libgfortran-ng=13.1.0=h69a702a_0
  - libgfortran5=13.1.0=h15d22d2_0
  - libglib=2.76.4=hebfc3b9_0
  - libgomp=13.1.0=he5830b7_0
  - libhwloc=2.9.1=hd6dc26d_0
  - libiconv=1.17=h166bdaf_0
  - libitk=5.3.0=hcedbc38_0
  - liblapack=3.9.0=17_linux64_openblas
  - libnghttp2=1.52.0=h61bc06f_0
  - libnsl=2.0.0=h7f98852_0
  - libopenblas=0.3.23=pthreads_h80387f5_0
  - libpng=1.6.39=h753d276_0
  - libsqlite=3.42.0=h2797004_0
  - libssh2=1.11.0=h0841786_0
  - libstdcxx-ng=13.1.0=hfd8a6a1_0
  - libtiff=4.4.0=h82bc61c_5
  - libuuid=2.38.1=h0b41bf4_0
  - libwebp-base=1.3.1=hd590300_0
  - libxcb=1.13=h7f98852_1004
  - libxml2=2.10.4=hfdac1af_0
  - libzlib=1.2.13=hd590300_5
  - libzopfli=1.0.3=h9c3ff4c_0
  - lz4-c=1.9.4=hcb278e6_0
  - msgpack-python=1.0.5=py39h4b4f3f3_0
  - ncurses=6.4=hcb278e6_0
  - networkx=3.1=pyhd8ed1ab_0
  - numcodecs=0.11.0=py39h227be39_1
  - openjpeg=2.5.0=h7d73246_1
  - openslide=3.4.1=h71beb9a_5
  - openslide-python=1.3.0=py39hd1e30aa_0
  - openssl=3.1.2=hd590300_0
  - packaging=23.1=pyhd8ed1ab_0
  - pcre2=10.40=hc3806b6_0
  - pillow=9.2.0=py39hf3a2cdf_3
  - pip=23.2.1=pyhd8ed1ab_0
  - pixman=0.40.0=h36c2ea0_0
  - platformdirs=3.10.0=pyhd8ed1ab_0
  - pooch=1.7.0=pyha770c72_3
  - pthread-stubs=0.4=h36c2ea0_1001
  - pysocks=1.7.1=pyha2e5f31_6
  - python=3.9.16=h2782a2a_0_cpython
  - python_abi=3.9=3_cp39
  - pywavelets=1.4.1=py39h389d5f1_0
  - readline=8.2=h8228510_1
  - requests=2.31.0=pyhd8ed1ab_0
  - scikit-image=0.21.0=py39h3d6467e_0
  - scikit-learn=1.3.0=py39hc236052_0
  - scipy=1.11.1=py39h6183b62_0
  - setuptools=68.0.0=pyhd8ed1ab_0
  - shapely=2.0.1=py39hf1c3bca_1
  - simpleitk=2.2.1=py39hdaa313e_1
  - snappy=1.1.10=h9fff704_0
  - sqlite=3.42.0=h2c6b66d_0
  - tbb=2021.9.0=hf52228f_0
  - threadpoolctl=3.2.0=pyha21a80b_0
  - tifffile=2022.10.10=pyhd8ed1ab_0
  - tiffslide=2.2.0=pyhd8ed1ab_0
  - tk=8.6.12=h27826a3_0
  - tzdata=2023c=h71feb2d_0
  - wheel=0.41.0=pyhd8ed1ab_0
  - xorg-kbproto=1.0.7=h7f98852_1002
  - xorg-libice=1.1.1=hd590300_0
  - xorg-libsm=1.2.4=h7391055_0
  - xorg-libx11=1.8.4=h0b41bf4_0
  - xorg-libxau=1.0.11=hd590300_0
  - xorg-libxdmcp=1.1.3=h7f98852_0
  - xorg-libxext=1.3.4=h0b41bf4_2
  - xorg-libxrender=0.9.10=h7f98852_1003
  - xorg-renderproto=0.11.1=h7f98852_1002
  - xorg-xextproto=7.3.0=h0b41bf4_1003
  - xorg-xproto=7.0.31=h7f98852_1007
  - xz=5.2.6=h166bdaf_0
  - zarr=2.16.1=pyhd8ed1ab_0
  - zfp=1.0.0=h27087fc_3
  - zlib=1.2.13=hd590300_5
  - zlib-ng=2.0.7=h0b41bf4_0
  - zstd=1.5.2=hfc55251_7
  - pip:
      - absl-py==1.4.0
      - array-record==0.4.1
      - astunparse==1.6.3
      - cachetools==5.3.1
      - click==8.1.6
      - dm-tree==0.1.8
      - etils==1.4.1
      - flatbuffers==23.5.26
      - gast==0.4.0
      - google-auth==2.22.0
      - google-auth-oauthlib==1.0.0
      - google-pasta==0.2.0
      - googleapis-common-protos==1.60.0
      - grpcio==1.56.2
      - h5py==3.9.0
      - importlib-metadata==6.8.0
      - importlib-resources==6.0.1
      - keras==2.13.1
      - keras-cv==0.6.1
      - libclang==16.0.6
      - markdown==3.4.4
      - markupsafe==2.1.3
      - numpy==1.24.3
      - nvidia-cublas-cu11==11.11.3.6
      - nvidia-cudnn-cu11==8.6.0.163
      - oauthlib==3.2.2
      - opt-einsum==3.3.0
      - promise==2.3
      - protobuf==3.20.3
      - psutil==5.9.5
      - pyasn1==0.5.0
      - pyasn1-modules==0.3.0
      - regex==2023.8.8
      - requests-oauthlib==1.3.1
      - rsa==4.9
      - six==1.16.0
      - tensorboard==2.13.0
      - tensorboard-data-server==0.7.1
      - tensorflow==2.13.0
      - tensorflow-datasets==4.9.2
      - tensorflow-estimator==2.13.0
      - tensorflow-io-gcs-filesystem==0.33.0
      - tensorflow-metadata==1.14.0
      - termcolor==2.3.0
      - toml==0.10.2
      - tqdm==4.66.1
      - typing-extensions==4.5.0
      - urllib3==1.26.16
      - werkzeug==2.3.6
      - wrapt==1.15.0
      - zipp==3.16.2
```
### Job submission on Biowulf
Example script to request a GPU, setup custom python virtual environment, and execute python code.
```bash
#! /bin/bash

#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=24:00:00

source your_conda_initalization_script
mamba activate tensorflow_vision

python your_deep_learning_magic_code.py $SLURM_JOB_ID
```
If the script is named *job.sh*, use the following command to submit a job: *sbatch job.sh*.  
The *your_conda_initialization_script* is the conda init file created in the [Custom Python Environment](https://github.com/yliu7366/yliu_utilities/tree/master/tutorials/biowulf#custom-python-environment) section.

### Keras simple MNIST convent example
A quick example for running convent models using TensorFlow/Keras can be found [here](https://keras.io/examples/vision/mnist_convnet/) once you have finished deep learning setup on Biowulf.
