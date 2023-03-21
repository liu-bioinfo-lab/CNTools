# CNTools

## System requirements
The software denpendencies are listed in `env.yml`. The required operating systems are Linux, MacOS, and Windows. The version the software has been tested on is v1.1.0.

## Installation guide
Create the conda environment by `conda env create -f env.yml`.

## Instructions for use

### Loading data
Usage:
```
python load.py [-h] --df_path DF_PATH --name NAME --out_dir OUT_DIR [--ct_order_path CT_ORDER_PATH]
```
Description of arguments can be accessed by `python load.py -h`.
```
Preprocess the tabular data and make them into a dictionary dataset.

required arguments:
  --df_path DF_PATH     input tabular data (.csv) path
  --name NAME           user-defined data name
  --out_dir OUT_DIR     output directory

optional arguments:
  --ct_order_path CT_ORDER_PATH
                        input CT order file (.json) path
```

### Identifying and smoothing cellular neighborhoods
Usage:
```
python identify.py [-h] --ds_path DS_PATH --out_dir OUT_DIR --n_cns N_CNS [--cns_path CNS_PATH] [--Naive s [n_neighbors ...]]
                   [--HMRF eps beta [max_neighbors max_iter max_iter_no_change ...]] [--seed SEED] [--verbose]
                   {CC,CFIDF,CNE,Spatial_LDA} ...
```
Description of general arguments for idenfication and smoothing can be accessed by `python identify.py -h`.
```
Identify and smooth CNs.

positional arguments:
  {CC,CFIDF,CNE,Spatial_LDA}
                        identification method

required arguments:
  --ds_path DS_PATH     input dataset (.pkl) path
  --out_dir OUT_DIR     output directory
  --n_cns N_CNS         number of CNs

optional arguments:
  --cns_path CNS_PATH   only do smoothing using the CN file (.pkl) at the given path
  --Naive s [n_neighbors ...]
                        Naive smoothing technique
                        s: minimum size of a CN instance
                        n_neighbors: effective only when input cell representions (feats) are None, number of nearest neighbors considered for building CC cell representations (default: 10)
  --HMRF eps beta [max_neighbors max_iter max_iter_no_change ...]
                        HMRF smoothing technique
                        eps: pixel radius of neighborhoods
                        beta: weight of each neighbor in the same CN
                        max_neighbors: maximum number of neighbors considered, -1 for all (default: -1)
                        max_iter: the maximum number of iterations (default: 50)
                        max_iter_no_chanage: stop if the loss does not change for some iterations (default: 3)
  --seed SEED           seed for reproducibility
  --verbose             whether to print out metric values
```
Description of specific arguments for each idenfication method can be accessed by `python identify.py <method> -h`. Using CNE as an example,
```
usage: identify.py CNE [-h] [--eta ETA] [--max_neighbors MAX_NEIGHBORS] [--exclude_cts [EXCLUDE_CTS [EXCLUDE_CTS ...]]]

required arguments:
  --eta ETA             scale parameter of the Gaussian distribution's std

optional arguments:
  --max_neighbors MAX_NEIGHBORS
                        maximum number of neighbors considered, -1 for all (default: -1)
  --exclude_cts [EXCLUDE_CTS [EXCLUDE_CTS ...]]
                        list of CTs to exclude in CN identification (default: [])
```

### Analyzing cellular neighborhoods
Run jupyter notebooks under the `analysis` folder.

## Demo
```
sh run_load.sh
sh run_idenfity.sh
```
Expected CN outputs and run time are in the `cn/*/CNE` folder. Expected analysis outputs are in the `analysis_res/*/CNE` folder.
