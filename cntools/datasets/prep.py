from .make_ds import Dataset
import json
import numpy as np


def load_ct_order(ct_order_path):
    return json.load(open(ct_order_path))['ct_order']


def prep_crc(df):
    df = df.rename(columns={'groups': 'Group', 'patients': 'Sample', 'File Name': 'Image', 'X:X': 'X', 'Y:Y': 'Y', 'ClusterName': 'CT'})
    df = df[df['CT'] != 'dirt']
    df['Group'] = df['Group'].apply(lambda r: ('CLR' if r == 1 else 'DII'))
    df['CT'] = df['CT'].apply(lambda r: r[0].upper() + r[1:])
    df = df[Dataset.KEY_COLS + ['CD4+ICOS+', 'CD4+Ki67+', 'CD4+PD-1+', 'CD8+ICOS+', 'CD8+Ki67+', 'CD8+PD-1+', 'Treg-ICOS+', 'Treg-Ki67+', 'Treg-PD-1+', 'CD68+CD163+ICOS+', 'CD68+CD163+Ki67+', 'CD68+CD163+PD-1+']]
    df = df.rename(columns={'CD68+CD163+ICOS+': 'Macs-ICOS+', 'CD68+CD163+Ki67+': 'Macs-Ki67+', 'CD68+CD163+PD-1+': 'Macs-PD-1+'})
    return df


def prep_crc_ori(df, no_dirt=False):
    df = df.rename(columns={'groups': 'Group', 'patients': 'Sample', 'File Name': 'Image', 'X:X': 'X', 'Y:Y': 'Y', 'ClusterName': 'CT'})
    if no_dirt:
        df = df[~df['neighborhood number final'].isna()]
        df = df[df['CT'] != 'dirt']
    df['Group'] = df['Group'].apply(lambda r: ('CLR' if r == 1 else 'DII'))
    df['CT'] = df['CT'].apply(lambda r: r[0].upper() + r[1:])
    cns = {}
    for sample, df_sample in df.groupby('Sample', sort=False):
        cns[sample] = {}
        for image, df_image in df_sample.groupby('Image', sort=False):
            if no_dirt:
                cns[sample][image] = df_image['neighborhood number final'].to_numpy(dtype=int) - 1
            else:
                cns[sample][image] = df_image['neighborhood10'].to_numpy(dtype=int)
    df = df[Dataset.KEY_COLS + ['CD4+ICOS+', 'CD4+Ki67+', 'CD4+PD-1+', 'CD8+ICOS+', 'CD8+Ki67+', 'CD8+PD-1+', 'Treg-ICOS+', 'Treg-Ki67+', 'Treg-PD-1+', 'CD68+CD163+ICOS+', 'CD68+CD163+Ki67+', 'CD68+CD163+PD-1+']]
    df = df.rename(columns={'CD68+CD163+ICOS+': 'Macs-ICOS+', 'CD68+CD163+Ki67+': 'Macs-Ki67+', 'CD68+CD163+PD-1+': 'Macs-PD-1+'})
    return df, cns


def prep_t2d(df):
    df = df.sort_values(by=['Group', 'Donor', 'Islet', 'Cell'])
    df['X'], df['Y'] = (df['XMin'] + df['XMax']) / 2, (df['YMin'] + df['YMax']) / 2
    df = df.drop(['XMin', 'XMax', 'YMin', 'YMax', 'Helper T cell', 'Cytotoxic T cell', 'Inactive T cell', 'M1 mac', 'M2 mac', 'M1/M2 mac', 'Other mac', 'HLA-DR+ EC', 'CD34+ EC', 'HLA-DR+ CD34+ EC'], axis=1)
    ct2nct = {'α cell': 'Alpha cells', 'δ cell': 'Delta cells', 'β cell': 'Beta cells', 'γ cell': 'Gamma cells', 'ɛ cell': 'Epsilon cells', 'T cell': 'T cells', 'Macrophage': 'Macrophages', 'Other immune cell': 'Other immune cells', 'EC': 'Endothelial cells', 'Pericyte': 'Pericytes'}
    df = df.rename(columns=ct2nct).rename(columns={'Donor': 'Sample', 'Islet': 'Image'})
    def annotate(row):
        for ct in ct2nct.values():
            if row[ct] == 1:
                return ct
        return np.nan
    df['CT'] = df.apply(lambda r: annotate(r), axis=1)
    df = df.dropna(subset=['CT'])
    df = df[Dataset.KEY_COLS + ['Arg1 Positive Classification', 'Ki67 Positive Classification']]
    return df


def prep_hlt(df):
    df = df.rename(columns={'sample': 'Image', 'x_in_file': 'X', 'y_in_file': 'Y', 'Cluster name': 'CT'})
    df = df[(df['Image'] != 'tonsil6677') & ((df['Image'] != 'tonsil8953') | ((df['X'] < 19200) & (df['Y'] > 4000)))]
    df = df[(df['CT'] != 'ECM near CD3+ cells') & (df['CT'] != 'ECM near CD45+ cells')]
    df[['CT']] = df[['CT']].replace({
        'Blood vessels near ECM': 'Blood vessels',
        'Epithelial cells near CD45+ cells': 'Epithelial cells',
        'Granulocytes near ECM': 'Granulocytes',
        'Macrophages near ECM': 'Macrophages',
        'Stromal cells near CD45+ cells': 'Stromal cells',
        'Activated T-Helper cells (CD278/CD4)': 'Activated T-helper cells (CD278/CD4)',
        'T Follicular Helper cells': 'T follicular helper cells',
        'T-Helper cells (CD4)': 'T-helper cells (CD4)',
        'T-Killer cells (CD8)': 'T-killer cells (CD8)'
    })
    df['Group'] = 0
    df['Sample'] = df['Image'].apply(lambda r: r if r == 'LN' or r == 'spleen' else 'tonsil')
    df = df[Dataset.KEY_COLS]
    return df
