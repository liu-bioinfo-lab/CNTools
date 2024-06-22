from .base import Base
from ..utils import cns_info
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from spatial_lda.featurization import featurize_samples
from spatial_lda.featurization import neighborhood_to_cluster
from spatial_lda.featurization import make_merged_difference_matrices
from spatial_lda.model import order_topics_consistently
import spatial_lda.model


class Spatial_LDA(Base):
    """A CN identification method.

    Parameters
    ----------
    n_cns : int
        Number of CNs.
    eps : float
        Pixel radius of neighborhoods.
    b : float
        Scale parameter of the Laplace distribution.
    train_size_fraction : float, default=0.99
        Fraction of training samples.
    n_processes : int, default=8 
        Number of parallel processes.
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

    def __init__(self, n_cns, eps, b, train_size_fraction=0.99, n_processes=8, exclude_cts=[], seed=0, verbose=True):
        super().__init__(n_cns, exclude_cts, seed, verbose)
        self.eps = eps
        self.b = b
        self.train_size_fraction = train_size_fraction
        self.n_processes = n_processes
    
    def fit(self, ds):
        data = ds.data
        dfs_image = {}
        for sample in data:
            for image in data[sample]:
                df_image = pd.DataFrame(data[sample][image].locs, columns=['X', 'Y'])
                df_image['cluster'] = data[sample][image].cts
                df_image['isb'] = True
                dfs_image[str(sample) + '-' + str(image)] = df_image
        feats = featurize_samples(dfs_image, neighborhood_to_cluster, self.eps, 'isb', 'X', 'Y', n_processes=self.n_processes, include_anchors=True)

        exclude_ids = [ds.ct_order.index(ct) for ct in self.exclude_cts]
        feats = np.delete(feats, exclude_ids, axis=1)

        all_sample_idxs = feats.index.map(lambda x: x[0])
        _sets = train_test_split(feats, test_size=1-self.train_size_fraction, stratify=all_sample_idxs)
        feats_image_train, _ = _sets
        train_difference_matrices = make_merged_difference_matrices(feats_image_train, dfs_image, 'X', 'Y')

        model = spatial_lda.model.train(sample_features=feats_image_train, 
                                        difference_matrices=train_difference_matrices,
                                        difference_penalty=self.b,
                                        n_topics=self.n_cns,
                                        n_parallel_processes=self.n_processes,                                                                         
                                        verbosity=1,
                                        admm_rho=0.1,
                                        primal_dual_mu=1e+5)
        order_topics_consistently([model])

        dfs_image_cns = pd.DataFrame(model.transform(feats), index=feats.index, columns=model.topic_weights.columns)
        cns = ds.get_data_tpl_copy()
        for sample_image in dfs_image:
            sample, image = sample_image.split('-')
            sample = int(sample) if sample.isnumeric() else sample
            image = int(image) if image.isnumeric() else image
            for i, _ in dfs_image[sample_image].iterrows():
                cns[sample][image].append(np.argmax(dfs_image_cns.loc(axis=0)[(sample_image, i),]))
        feats = feats.reindex(sorted(feats.columns), axis=1).to_numpy()
        
        self.feats = feats
        if self.verbose:
            cns_info(ds.data, self.n_cns, cns)
        return cns
