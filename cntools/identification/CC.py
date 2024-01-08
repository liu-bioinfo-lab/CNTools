from .base import Base
from ..utils import cns_info
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


class CC(Base):
    """A CN identification method.

    Parameters
    ----------
    n_cns : int
        Number of CNs.
    m : int
        Number of neighbors.
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

    def __init__(self, n_cns, m, exclude_cts=[], seed=0, verbose=True):
        super().__init__(n_cns, exclude_cts, seed, verbose)
        self.m = m
    
    def fit(self, ds):
        data = ds.data
        feats = []
        for sample in tqdm(data):
            for image in tqdm(data[sample], leave=False):
                locs = data[sample][image].locs
                n_neighbors_real = self.m if data[sample][image].n_cells >= self.m else data[sample][image].n_cells
                _, indices = NearestNeighbors(n_neighbors=n_neighbors_real).fit(locs).kneighbors(locs)
                for nbs in indices:
                    feats.append(data[sample][image].cts_oh[nbs].sum(axis=0))
        feats = np.array(feats)

        exclude_ids = [ds.ct_order.index(ct) for ct in self.exclude_cts]
        feats = np.delete(feats, exclude_ids, axis=1)
        
        feats = normalize(feats, norm='l1', axis=1)
        cns = ds.flat_to_dic(MiniBatchKMeans(n_clusters=self.n_cns, random_state=self.seed).fit_predict(feats))
        
        self.feats = feats
        if self.verbose:
            cns_info(ds, self.n_cns, cns)
        return cns
