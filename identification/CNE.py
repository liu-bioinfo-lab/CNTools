import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm


def CNE(ds, n_cns, eta, max_neighbors, exclude_cts, seed):
    data = ds.data
    feats = []
    for sample in tqdm(data):
        for image in tqdm(data[sample], leave=False):
            print(sample, image)
            locs = data[sample][image].locs
            if max_neighbors == -1:
                pdists = pairwise_distances(locs)
                min_neighbor_dists = (pdists + 1e12 * (pdists == 0)).min(axis=1, keepdims=True)
                pdists += (pdists == 0) * min_neighbor_dists
                prob = np.exp(-pdists ** 2 / ((eta * min_neighbor_dists) ** 2))
                feats.append(prob @ data[sample][image].cts_oh)
            else:
                pdists, indices = NearestNeighbors(n_neighbors=max_neighbors).fit(locs).kneighbors(locs)
                min_neighbor_dists = (pdists + 1e12 * (pdists == 0)).min(axis=1, keepdims=True)
                pdists += (pdists == 0) * min_neighbor_dists
                prob = np.exp(-pdists ** 2 / ((eta * min_neighbor_dists) ** 2))
                feats.append((np.expand_dims(prob, axis=1) @ data[sample][image].cts_oh[indices]).squeeze(axis=1))
    feats = np.concatenate(feats)
    
    exclude_ids = [ds.ct_order.index(ct) for ct in exclude_cts]
    feats = np.delete(feats, exclude_ids, axis=1)
    ct_counts = np.delete(ds.ct_counts, exclude_ids, axis=0)
    
    feats = normalize(feats, norm='l1', axis=1) * np.log(ct_counts.sum() / (ct_counts + 1))

    cns = ds.flat_to_dic(KMeans(n_clusters=n_cns, random_state=seed).fit_predict(feats))
    return cns, feats
