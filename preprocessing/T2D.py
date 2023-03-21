import numpy as np


def preprocess(df, cols):
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
    df = df[cols + ['Arg1 Positive Classification', 'Ki67 Positive Classification']]
    return df
