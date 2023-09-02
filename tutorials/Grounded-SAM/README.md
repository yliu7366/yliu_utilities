# Grounded-SAM Experiments

## Local Installation
Read this section if you need a little bit more clarification about the local installation steps. The Grounding-DINO module requires a local CUDA installation so a default [pytorch and CUDA installation](https://pytorch.org/get-started/locally/) in conda won't work. Download the CUDA library installation file from [Nvidia](https://developer.nvidia.com/cuda-toolkit-archive) and install it to a local folder so it won't mess up system-wide libraries. I choosed to match the download CUDA library version with pytorch CUDA version, which is 11.7 following the official pytorch installation guide as of 09/02/2023. Now export the CUDA_HOME environment variable using the CUDA installation folder and follow the Grounded-SAM local installation steps.

### Rust and Cargo
The Tag2Text submodule requires rust compiler. Install rust in *Ubuntu 22.04* by running the script below.
```bash
sudo apt install rustc cargo
```
