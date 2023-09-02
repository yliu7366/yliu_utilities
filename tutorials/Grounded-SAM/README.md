# Grounded-SAM Experiments

## Local Installation
Read this section if you need a little bit more clarification about the local installation steps. The Grounding-DINO module requires a local CUDA installation so a standard conda pytorch and CUDA installation won't work. Download the CUDA library installation file from Nvidia and install it to a local folder so it won't mess up system-wide libraries. I choosed to match the download CUDA library version with pytorch CUDA version, which is 11.7 as of 09/02/2023. Now export the CUDA_HOME environment variable using the CUDA installation folder and follow the Grounded-SAM local installation steps.
