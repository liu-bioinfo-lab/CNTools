gdown --folder https://drive.google.com/drive/folders/1eC3ahSIzCZ5vq_bynSl3IX7V9w_bQgg8?usp=share_link
python load.py --df_path data/CRC/CRC_clusters_neighborhoods_markers.csv --name CRC --out_dir data/CRC --ct_order_path data/CRC/ct_order.json
python load.py --df_path data/T2D/Cell-ID_by-islet.csv --name T2D --out_dir data/T2D --ct_order_path data/T2D/ct_order.json
python load.py --df_path data/HLT/Lymphoid_Data.csv --name HLT --out_dir data/HLT