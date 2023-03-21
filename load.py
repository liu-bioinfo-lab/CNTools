import argparse
import pandas as pd
import json
from dataset import Dataset
import pickle
import os
from preprocessing import *


def load(df_path, name, out_dir, ct_order_path):
    df = pd.read_csv(df_path)
    try:
        print(f'Preprocessing {name}...')
        df = eval(name).preprocess(df, ['Group', 'Sample', 'Image', 'X', 'Y', 'CT'])
    except NameError:
        pass
    ct_order = json.load(open(ct_order_path))['ct_order'] if ct_order_path else None
    ds = Dataset(df, name, ct_order)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(os.path.join(out_dir, f'{name}_df.csv'), index=False)
    pickle.dump(ds, open(os.path.join(out_dir, f'{name}_ds.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the tabular data and make them into a dictionary dataset.', formatter_class=argparse.RawTextHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--df_path', type=str, help='input tabular data (.csv) path', required=True)
    required.add_argument('--name', type=str, help='user-defined data name', required=True)
    required.add_argument('--out_dir', type=str, help='output directory', required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--ct_order_path', type=str, help='input CT order file (.json) path')

    args = parser.parse_args()
    load(**vars(args))
