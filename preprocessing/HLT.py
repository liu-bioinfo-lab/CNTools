def preprocess(df, cols):
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
    df = df[cols]
    return df
