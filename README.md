# CNTools

## System requirements
The software denpendencies are listed in `pyproject.toml`. The software is independent of operating systems. The version the software has been tested on is v2.0.3.

## Installation guide
As we need a conda package pydot=1.4.2 (not a pip one), the package should be installed by
```
conda create -n cntools python=3.8 pydot=1.4.2
python -m pip install cntools
```

## Instructions for use

### Idenfity and smooth cellular neighborhoods
See `tests/test_crc.ipynb` for CRC dataset, `tests/test_t2d.ipynb` for T2D dataset, and `tests/test_hlt.ipynb` for HLT dataset.

### Analyze cellular neighborhoods
See jupyter notebooks in the `tests/analysis` folder.

## Demo
Run `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb` for CN identification and smoothing. Run jupyter notebooks in the `tests/analysis/` folder for CN analyses. Expected CN outputs are in the `tests/cn/*/CNE/` folder. Expected analysis outputs are in the `tests/analysis_res/*/CNE/` folder.

## Acknowledgements
Our implementation adapts the code of [Spatial LDA](https://github.com/calico/spatial_lda), [Schurch et al. (2020)](https://github.com/nolanlab/NeighborhoodCoordination), and [Bhate et al. (2022)](https://github.com/nolanlab/TissueSchematics) as cellular neighborhood identification and analysis methods. We thank the authors for sharing their code.