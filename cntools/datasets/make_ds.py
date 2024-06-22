import numpy as np
from scipy.spatial import Delaunay
from sklearn.preprocessing import label_binarize
from collections import Counter
from copy import deepcopy


class Dataset():
    KEY_COLS = ['Group', 'Sample', 'Image', 'X', 'Y', 'CT']
    
    def __init__(self, df, ct_order): # Table: Group, Sample, Image, X, Y, CT
        self.ct_order = ct_order if ct_order else sorted(list(df['CT'].unique()))
        self.n_cells = len(df)
        self.n_cts = len(self.ct_order)
        self.group2sample, self.sample2group, self.data, self.data_tpl, self.ct_counts = self.make_data(df)
        self.cts_sg_to_oh()

    def make_data(self, df):
        group2sample = {k: list(v) for k, v in df[['Sample', 'Group']].drop_duplicates().groupby('Group')['Sample'].unique().to_dict().items()}
        sample2group = df.set_index('Sample')['Group'].to_dict()
        data, data_tpl = {}, {}
        ct_counts = np.zeros(self.n_cts, dtype=int)
        for sample, df_sample in df.groupby('Sample', sort=False):
            data[sample], data_tpl[sample] = {}, {}
            for image, df_image in df_sample.groupby('Image', sort=False):
                data[sample][image], data_tpl[sample][image] = self.make_image(df_image), []
                for ct, ct_count in Counter(data[sample][image].cts).items():
                    ct_counts[ct] += ct_count
        return group2sample, sample2group, data, data_tpl, ct_counts

    class Image():
        def __init__(self, locs, cts, edge_indices):
            self.locs = locs
            self.cts = cts
            self.edge_indices = edge_indices
            self.n_cells = len(locs)
            self.cts_oh = None

    def make_image(self, df_image):
        locs, cts = df_image[['X', 'Y']].to_numpy(), df_image['CT'].apply(lambda r: self.ct_order.index(r)).to_numpy()
        tri = Delaunay(locs)
        nbid, nbs = tri.vertex_neighbor_vertices
        edge_indices = []
        for i in range(len(df_image)):
            edge_indices += [(e1, e2) for e1, e2 in zip([i for _ in range(nbid[i+1] - nbid[i])], nbs[nbid[i]:nbid[i+1]])]
        return self.Image(locs, cts, edge_indices)

    def cts_sg_to_oh(self):
        for sample in self.data:
            for image in self.data[sample]:
                self.data[sample][image].cts_oh = label_binarize(self.data[sample][image].cts, classes=range(self.n_cts))

    def get_data_tpl_copy(self):
        return deepcopy(self.data_tpl)

    def flat_to_dic(self, data_flat):
        data = self.get_data_tpl_copy()
        i = 0
        for sample in data:
            for image in data[sample]:
                data[sample][image] = data_flat[i:i+self.data[sample][image].n_cells]
                i += self.data[sample][image].n_cells
        return data
    
    def get_cts_oh(self):
        cts_oh = self.get_data_tpl_copy()
        for sample in self.data:
            for image in self.data[sample]:
                cts_oh[sample][image] = self.data[sample][image].cts_oh
        return cts_oh


def cns_sg_to_oh(cns, n_cns):
    cns_oh = {}
    for sample in cns:
        cns_oh[sample] = {}
        for image in cns[sample]:
            cns_oh[sample][image] = label_binarize(cns[sample][image], classes=range(n_cns))
    return cns_oh


def cns_oh_to_sg(cns_oh):
    cns = {}
    for sample in cns_oh:
        cns[sample] = {}
        for image in cns_oh[sample]:
            cns[sample][image] = cns_oh[sample][image].nonzero()[1]
    return cns


def dic_to_flat(data):
    data_flat = []
    for sample in data:
        for image in data[sample]:
            data_flat.append(data[sample][image])
    data_flat = np.concatenate(data_flat)
    return data_flat
