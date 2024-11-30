# SMILE

## Overview

SMILE is designed for alignment and integration of spatially resolved transcriptomics data.


## Installation
The SMILE package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework.

It's recommended to create a separate conda environment for running SMILE:

```
#create an environment called env_SMILE
conda create -n env_SMILE python=3.11

#activate your environment
conda activate env_SMILE
```
First clone the repository. 

```
git clone https://github.com/zhanglhwhu/SMILE.git
cd SMILE-main
```

Install all the required packages.

```
pip install -r requiements.txt
```
The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

The torch-geometric library is also required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation

Install SMILE.

```
pip install git+git://github.com/zhanglhwhu/SMILE.git
```



## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder to show how to use SMILE. 

- Tutorial 1: Integrating simulation data
- Tutorial 2: Integrating DLPFC slices 

## Support

If you have any questions, please feel free to contact us [zhanglh@whu.edu.cn](mailto:zhanglh@whu.edu.cn). 


