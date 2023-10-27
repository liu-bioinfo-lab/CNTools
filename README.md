# CNTools

## System requirements
The software denpendencies are listed in `env.yml` and `pyproject.toml`. The required operating systems are Linux, MacOS, and Windows. The version the software has been tested on is v2.0.0.

## Installation guide
```
conda env create -f env.yml
conda activate CNTools
pip install -e .
```

## Instructions for use

### Idenfity and smooth cellular neighborhoods
See `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb`.

### Analyze cellular neighborhoods
See jupyter notebooks in the `tests/analysis` folder.

## Demo
Run `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb` for CN identification and smoothing. Run jupyter notebooks in the `tests/analysis` folder for CN analyses. Expected CN outputs are in the `cn/*/CNE` folder. Expected analysis outputs are in the `analysis_res/*/CNE` folder.

## Acknowledgements
Our implementation adapts the code of [Spatial LDA](https://github.com/calico/spatial_lda), [Schurch et al. (2020)](https://github.com/nolanlab/NeighborhoodCoordination), and [Bhate et al. (2022)](https://github.com/nolanlab/TissueSchematics) as cellular neighborhood identification and analysis methods. We thank the authors for sharing their code.
