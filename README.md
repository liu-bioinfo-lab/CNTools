# CNTools

## System requirements
The software denpendencies are listed in `pyproject.toml`. The software is independent of operating systems. The version the software has been tested on is v2.0.8.

## Installation guide
As we need a conda package pydot=1.4.2 (not a pip one), the package should be installed by
```
conda create -n cntools python=3.8 pydot=1.4.2
python -m pip install cntools
```

## Instructions for use

### Idenfity and smooth cellular neighborhoods
See `tests/test_crc.ipynb` for CRC dataset, `tests/test_t2d.ipynb` for T2D dataset, and `tests/test_hlt.ipynb` for HLT dataset. For example, to run CNE on the CRC dataset, just do
```
from cntools.datasets import load_ct_order, prep_crc, Dataset
from cntools.identification import CNE
from cntools.smoothing import NaiveSmooth

# load dataset
df = prep_crc(pd.read_csv('data/CRC/CRC_clusters_neighborhoods_markers.csv'))
ct_order = load_ct_order('data/CRC/ct_order.json')
ds = Dataset(df, ct_order)

# identify
identifier = CNE(n_cns=9, perp=15, lam=0.25)
cns = identifier.fit(ds) # output CN

# smooth
smoother = NaiveSmooth(ds=ds, n_cns=identifier.n_cns, feats=identifier.feats, s=3)
cns_smoothed_naive = smoother.fit(cns) # output CN after smoothing
```

### Analyze cellular neighborhoods
See jupyter notebooks in the `tests/analysis` folder.

## Demo
Run `tests/test_crc.ipynb`, `tests/test_t2d.ipynb`, and `tests/test_hlt.ipynb` for CN identification and smoothing. Run jupyter notebooks in the `tests/analysis/` folder for CN analyses. Expected CN outputs are in the `tests/cn/*/CNE/` folder. Expected analysis outputs are in the `tests/analysis_res/*/CNE/` folder.

## Acknowledgements
Our implementation adapts the code of [Spatial LDA](https://github.com/calico/spatial_lda), [Schurch et al. (2020)](https://github.com/nolanlab/NeighborhoodCoordination), and [Bhate et al. (2022)](https://github.com/nolanlab/TissueSchematics) as cellular neighborhood identification and analysis methods. We thank the authors for sharing their code.
```
@article{chen2020modeling,
  title={Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments},
  author={Chen, Zhenghao and Soifer, Ilya and Hilton, Hugo and Keren, Leeat and Jojic, Vladimir},
  journal={Journal of Computational Biology},
  year={2020},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}

@article{schurch2020coordinated,
  title={Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front},
  author={Sch{\"u}rch, Christian M and Bhate, Salil S and Barlow, Graham L and Phillips, Darci J and Noti, Luca and Zlobec, Inti and Chu, Pauline and Black, Sarah and Demeter, Janos and McIlwain, David R and others},
  journal={Cell},
  volume={182},
  number={5},
  pages={1341--1359},
  year={2020},
  publisher={Elsevier}
}

@article{bhate2022tissue,
  title={Tissue schematics map the specialization of immune tissue motifs and their appropriation by tumors},
  author={Bhate, Salil S and Barlow, Graham L and Sch{\"u}rch, Christian M and Nolan, Garry P},
  journal={Cell Systems},
  volume={13},
  number={2},
  pages={109--130},
  year={2022},
  publisher={Elsevier}
}
```
