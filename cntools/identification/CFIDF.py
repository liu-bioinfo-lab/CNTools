from .base import Base
from ..utils import cns_info
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import normalize
from tqdm import tqdm


class CFIDF(Base):
    """A CN identification method.

    Parameters
    ----------
    n_cns : int
        Number of CNs.
    eps : float
        Pixel radius of neighborhoods.
    r : flost
        Resolution of Louvain algorithm.
    max_neighbors : int, default=-1
        Maximum number of neighbors considered, -1 for all.
    exclude_cts : list, default=[]
        List of CTs to exclude in CN identification.
    seed : int, default=0
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print CN entropy and mean instance size.
    
    Attributes
    ----------
    feats: ndarray of shape (n_cells, n_features)
        Intermediate features for each cell.
    """

    def __init__(self, n_cns, eps, r, max_neighbors=-1, exclude_cts=[], seed=0, verbose=True):
        super().__init__(n_cns, exclude_cts, seed, verbose)
        self.eps = eps
        self.r = r
        self.max_neighbors = max_neighbors

    def fit(self, ds):
        data = ds.data
        nb_cell, feats_nb = ds.get_data_tpl_copy(), ds.get_data_tpl_copy()

        exclude_ids = [ds.ct_order.index(ct) for ct in self.exclude_cts]
        ct_counts = np.delete(ds.ct_counts, exclude_ids, axis=0)

        for sample in tqdm(data):
            for image in tqdm(data[sample], leave=False):
                nb_cell[sample][image], feats_nb[sample][image] = {}, {}
                locs = data[sample][image].locs
                if self.max_neighbors == -1:
                    knn_graph = radius_neighbors_graph(locs, self.eps)
                    g = nx.from_scipy_sparse_matrix(knn_graph)
                    weights_mat = pairwise_distances(locs) * knn_graph.toarray()
                    edge_weights = np.log2(1 / (0.005 + weights_mat / ((weights_mat ** 2).sum() / 2) ** 0.5))
                    for e in g.edges:
                        g[e[0]][e[1]]['weight'] = edge_weights[e[0]][e[1]]
                else:
                    g = nx.Graph()
                    pdists, indices = NearestNeighbors(n_neighbors=self.max_neighbors).fit(locs).kneighbors(locs)
                    adj = (pdists <= self.eps)
                    weights_mat = pdists * adj
                    edge_weights = np.log2(1 / (0.005 + weights_mat / ((weights_mat ** 2).sum() / 2) ** 0.5))
                    for i in range(len(indices)):
                        for j in range(len(indices[0])):
                            if adj[i][j] and i < indices[i][j]:
                                g.add_edge(i, indices[i][j], weight=edge_weights[i][j])                            
                partition = community_louvain.best_partition(g, resolution=self.r, random_state=self.seed)
                for i in partition:
                    if partition[i] not in nb_cell[sample][image]:
                        nb_cell[sample][image][partition[i]] = []
                    nb_cell[sample][image][partition[i]].append(i)
                for i in nb_cell[sample][image]:
                    feats_nb[sample][image][i] = data[sample][image].cts_oh[nb_cell[sample][image][i]].sum(axis=0)
        feats_nb_normed = []
        for sample in data:
            for image in data[sample]:
                for nb_id in feats_nb[sample][image]:
                    feats_nb[sample][image][nb_id] = np.delete(feats_nb[sample][image][nb_id], exclude_ids, axis=0)
                    feats_nb[sample][image][nb_id] = normalize([feats_nb[sample][image][nb_id]], norm='l1', axis=1)[0]
                    feats_nb[sample][image][nb_id] *= np.log(ct_counts.sum() / (ct_counts + 1))
                    feats_nb_normed.append(feats_nb[sample][image][nb_id])
        feats_nb_normed = np.array(feats_nb_normed)
        cns_nb_flat = KMeans(n_clusters=self.n_cns, random_state=self.seed).fit_predict(feats_nb_normed)

        cns = ds.get_data_tpl_copy()
        feats_normed = []
        i = 0
        for sample in data:
            for image in data[sample]:
                feats_normed_tmp = np.zeros(np.delete(data[sample][image].cts_oh, exclude_ids, axis=1).shape)
                cns[sample][image] = np.zeros(data[sample][image].n_cells, dtype='int')
                for nb_id, nb in nb_cell[sample][image].items():
                    cns[sample][image][nb] = cns_nb_flat[i]
                    feats_normed_tmp[nb] = feats_nb[sample][image][nb_id]
                    i += 1
                feats_normed.append(feats_normed_tmp)
        feats_normed = np.concatenate(feats_normed)

        self.feats = feats_normed
        if self.verbose:
            cns_info(ds.data, self.n_cns, cns)
        return cns
