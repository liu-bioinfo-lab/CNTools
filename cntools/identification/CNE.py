from .base import Base
from ..utils import cns_info, delauney_edges, KMeans_reg
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm


class CNE(Base):
    """A CN identification method.

    Parameters
    ----------
    n_cns : int
        Number of CNs.
    perp : float
        Perplexity measure to assign different weights to neighboring cells.
    lam : float, default=0.25
        Coefficient of the spatial regularizer for k-means.
    max_neighbors : int, default=30
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

    def __init__(self, n_cns, perp, lam=0.25, max_neighbors=30, exclude_cts=[], seed=0, verbose=True):
        super().__init__(n_cns, exclude_cts, seed, verbose)
        self.perp = perp
        self.lam = lam
        self.max_neighbors = max_neighbors
    
    def fit(self, ds):
        data = ds.data
        feats = []
        for sample in tqdm(data):
            for image in tqdm(data[sample], leave=False):
                locs = data[sample][image].locs
                if self.max_neighbors == -1:
                    pdists = pairwise_distances(locs)
                    min_neighbor_dists = (pdists + 1e12 * (pdists == 0)).min(axis=1, keepdims=True)
                    pdists += (pdists == 0) * min_neighbor_dists
                    prob = _binary_search_perplexity(np.array(pdists ** 2, dtype=np.float32), min(self.perp, len(pdists)), 0)
                    feats.append(prob @ data[sample][image].cts_oh)
                else:
                    n_neighbors_eff = min(self.max_neighbors, data[sample][image].n_cells)
                    pdists, indices = NearestNeighbors(n_neighbors=n_neighbors_eff).fit(locs).kneighbors(locs)
                    min_neighbor_dists = (pdists + 1e12 * (pdists == 0)).min(axis=1, keepdims=True)
                    pdists += (pdists == 0) * min_neighbor_dists
                    prob = _binary_search_perplexity(np.array(pdists ** 2, dtype=np.float32), min(self.perp, len(pdists)), 0)
                    feats.append((np.expand_dims(prob, axis=1) @ data[sample][image].cts_oh[indices]).squeeze(axis=1))
        feats = np.concatenate(feats)
        
        exclude_ids = [ds.ct_order.index(ct) for ct in self.exclude_cts]
        feats = np.delete(feats, exclude_ids, axis=1)
        ct_counts = np.delete(ds.ct_counts, exclude_ids, axis=0)
        
        feats = normalize(feats, norm='l1', axis=1) * np.log(ct_counts.sum() / (ct_counts + 1))
        cns = ds.flat_to_dic(KMeans_reg(n_clusters=self.n_cns, random_state=self.seed, lam=self.lam, edges=delauney_edges(ds)).fit_predict(feats))

        self.feats = feats
        if self.verbose:
            cns_info(ds, self.n_cns, cns)
        return cns
