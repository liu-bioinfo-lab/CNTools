from .base import Base
from ..utils import cns_info
import numpy as np
import networkx as nx
from networkx import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy


def remove_small(data_image, cns_image, feats_image, s):
    cns_image = deepcopy(cns_image)
    graph = nx.Graph(data_image.edge_indices)
    for edge in graph.edges:
        if cns_image[edge[0]] != cns_image[edge[1]]:
            graph.remove_edge(edge[0], edge[1])
    small_nbs, large_nbs = set(), set()
    for i in connected_components(graph):
        if len(i) < s:
            small_nbs = small_nbs.union(i)
        else:
            large_nbs = large_nbs.union(i)
    if not large_nbs:
        return cns_image
    graph = nx.Graph(data_image.edge_indices)
    while small_nbs:
        large_nbs_added = set()
        small_nbs_left = set()
        for cell in small_nbs:
            closest_nb, simi = None, -2
            if not [c for c in graph.neighbors(cell)]:
                large_nbs_added.add(cell)
                continue
            for cell_nb in graph.neighbors(cell):
                if cell_nb in large_nbs:
                    simi_curr = cosine_similarity(feats_image[cell:cell+1], feats_image[cell_nb:cell_nb+1])
                    if simi_curr > simi:
                        closest_nb, simi = cell_nb, simi_curr
            if simi > -2:
                large_nbs_added.add(cell)
                cns_image[cell] = cns_image[closest_nb]
            else:
                small_nbs_left.add(cell)      
        large_nbs = large_nbs.union(large_nbs_added)
        small_nbs = small_nbs_left
    return cns_image


class NaiveSmooth(Base):
    """A CN smoothing technique.

    Parameters
    ----------
    ds : Dataset
        Dataset object for the input dataset.
    n_cns : int
        Number of CNs.
    feats : ndarray of shape (n_cells, n_features)
        Intermediate features for each cell.
    s : int, default=3
        Minimum size of a CN instance.
    n_neighbors : int, default=10
        Effective only when input cell representions (feats) are None, number of nearest neighbors considered for building CC cell representations.
    verbose : bool, default=True
        Whether to print CN entropy and mean instance size.
    """

    def __init__(self, ds, n_cns, feats, s=3, n_neighbors=10, verbose=True):
        super().__init__(ds, n_cns, verbose)
        self.feats = feats
        self.s = s
        self.n_neighbors = n_neighbors
    
    def fit(self, cns):
        data = self.ds.data
        cns_smoothed = self.ds.get_data_tpl_copy()
        if self.feats is not None:
            i = 0
            for sample in data:
                for image in data[sample]:
                    cns_smoothed[sample][image] = remove_small(data[sample][image], cns[sample][image], self.feats[i:i+data[sample][image].n_cells], self.s)
                    i += data[sample][image].n_cells
        else:
            for sample in data:
                for image in data[sample]:
                    feats_image = []
                    locs = data[sample][image].locs
                    n_neighbors_real = self.n_neighbors if data[sample][image].n_cells >= self.n_neighbors else data[sample][image].n_cells
                    _, indices = NearestNeighbors(n_neighbors=n_neighbors_real).fit(locs).kneighbors(locs)
                    for nbs in indices:
                        feats_image.append(data[sample][image].cts_oh[nbs].mean(axis=0))
                    feats_image = np.array(feats_image)
                    cns_smoothed[sample][image] = remove_small(data[sample][image], cns[sample][image], feats_image, self.s)
        if self.verbose:
            cns_info(self.ds, self.n_cns, cns_smoothed)
        return cns_smoothed
