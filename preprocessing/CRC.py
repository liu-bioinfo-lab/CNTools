def preprocess(df, cols):
    df = df.rename(columns={'groups': 'Group', 'patients': 'Sample', 'File Name': 'Image', 'X:X': 'X', 'Y:Y': 'Y', 'ClusterName': 'CT'})
    df = df[df['CT'] != 'dirt']
    df['Group'] = df['Group'].apply(lambda r: ('CLR' if r == 1 else 'DII'))
    df['CT'] = df['CT'].apply(lambda r: r[0].upper() + r[1:])
    df = df[cols + ['CD4+ICOS+', 'CD4+Ki67+', 'CD4+PD-1+', 'CD8+ICOS+', 'CD8+Ki67+', 'CD8+PD-1+', 'Treg-ICOS+', 'Treg-Ki67+', 'Treg-PD-1+']]
    return df
