# CNTools

## System requirements
The software denpendencies are listed in `env.yml` and `pyproject.toml`. The required operating systems are Linux, MacOS, and Windows. The version the software has been tested on is v2.0.0.

## Installation guide
Create the conda environment by `conda env create -f env.yml`, and then install CNTools package by `pip install .`

## Instructions for use

### Idenfity and smooth cellular neighborhoods
See `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb`.

### Analyze cellular neighborhoods
See jupyter notebooks in the `tests/analysis` folder.

## Demo
Run `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb` for CN identification and smoothing. Run jupyter notebooks in the `tests/analysis` folder for CN analyses. Expected CN outputs are in the `cn/*/CNE` folder. Expected analysis outputs are in the `analysis_res/*/CNE` folder.
