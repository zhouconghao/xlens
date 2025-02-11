# xlens (Graviational Lensing from Image Pixels)

[![tests](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml/badge.svg)](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml)
[![conda-forge](https://anaconda.org/conda-forge/xlens/badges/version.svg)](https://anaconda.org/conda-forge/xlens)


## Installation

### Github
Users can clone this repository and install the latest package by
```shell
git clone https://github.com/mr-superonion/xlens.git
cd xlens
# install required softwares
conda install -c conda-forge --file requirements.txt
# install required softwares for unit tests (if necessary)
conda install -c conda-forge --file requirements_test.txt
pip install . --user
```

### Conda-forge
stable verion can be installed from conda-forge
```
conda install -c conda-forge xlens
```

### Input Galaxy Catalog
One can download and setup the input galaxy catalog that is generated by catsim
in the  WeakLensingDeblending package

```shell
wget https://idark.ipmu.jp/~xiangchong.li/Research/image_simulation/catsim-v4.tar.gz

tar xvfz catsim-v4.tar.gz
export CATSIM_DIR=/path/to/catsim

```

