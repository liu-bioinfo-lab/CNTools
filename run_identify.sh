# CRC
# python identify.py --ds_path data/CRC/CRC_ds.pkl --out_dir cn/CRC/CC --n_cns 10 --cns_path cn/CRC/CC/cns_original.pkl --Naive 3 --HMRF 40 9 --verbose
# python identify.py --ds_path data/CRC/CRC_ds.pkl --out_dir cn/CRC/CFIDF --n_cns 9 --Naive 3 --HMRF 40 9 --seed 0 --verbose CFIDF --eps 32 --r 0.8
python identify.py --ds_path data/CRC/CRC_ds.pkl --out_dir cn/CRC/CNE --n_cns 9 --Naive 3 --HMRF 40 9 --seed 0 --verbose CNE --eta 2
# python identify.py --ds_path data/CRC/CRC_ds.pkl --out_dir cn/CRC/Spatial_LDA --n_cns 9 --Naive 3 --HMRF 40 9 --verbose Spatial_LDA --eps 50 --b 0.025

# T2D
# python identify.py --ds_path data/T2D/T2D_ds.pkl --out_dir cn/T2D/CC --n_cns 6 --Naive 3 --HMRF 45 9 --seed 0 --verbose CC --m 5
# python identify.py --ds_path data/T2D/T2D_ds.pkl --out_dir cn/T2D/CFIDF --n_cns 6 --cns_path cn/T2D/CFIDF/cns_original.pkl --Naive 3 --HMRF 45 9 --verbose
python identify.py --ds_path data/T2D/T2D_ds.pkl --out_dir cn/T2D/CNE --n_cns 6 --Naive 3 --HMRF 45 9 --seed 0 --verbose CNE --eta 2
# python identify.py --ds_path data/T2D/T2D_ds.pkl --out_dir cn/T2D/Spatial_LDA --n_cns 6 --Naive 3 --HMRF 45 9 --verbose Spatial_LDA --eps 100 --b 0.25 --train_size_fraction 0.989

# HLT
# python identify.py --ds_path data/HLT/HLT_ds.pkl --out_dir cn/HLT/CC --n_cns 11 --Naive 3 --HMRF 29 9 100 --seed 0 --verbose CC --m 20
# python identify.py --ds_path data/HLT/HLT_ds.pkl --out_dir cn/HLT/CFIDF --n_cns 11 --Naive 3 --HMRF 29 9 100 --seed 0 --verbose CFIDF --eps 23 --r 0.8 --max_neighbors 100
python identify.py --ds_path data/HLT/HLT_ds.pkl --out_dir cn/HLT/CNE --n_cns 11 --Naive 3 --HMRF 29 9 100 --seed 0 --verbose CNE --eta 2 --max_neighbors 100
# python identify.py --ds_path data/HLT/HLT_ds.pkl --out_dir cn/HLT/Spatial_LDA --n_cns 11 --Naive 3 --HMRF 29 9 100 --verbose Spatial_LDA --eps 50 --b 0.025
