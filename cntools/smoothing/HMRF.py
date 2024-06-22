from .base import Base
from ..datasets import cns_sg_to_oh, cns_oh_to_sg, dic_to_flat
from ..utils import cns_info
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def count_neighbors(ds, eps, include_neighbors=False, n_included=100):
    data = ds.data
    n_neighbors = 0
    for sample in data:
        for image in data[sample]:
            locs = data[sample][image].locs
            pdists = pairwise_distances(locs) if not include_neighbors else NearestNeighbors(n_neighbors=n_included).fit(locs).kneighbors(locs)[0]
            n_neighbors += (pdists <= eps).sum()
    return n_neighbors / ds.n_cells - 1


def cal_prob_ci_cni(cns_oh_neighbor, beta):
    logits = np.exp(beta * cns_oh_neighbor.sum(axis=0))
    return logits / logits.sum()


def cal_prob_xi_ci(cts_oh_flat, cns_oh_flat):
    prob_xi_ci = np.zeros((cts_oh_flat.shape[1], cns_oh_flat.shape[1])) + 1e-12
    for i in range(cns_oh_flat.shape[1]):
        prob_xi_ci[:, i] += (cts_oh_flat[cns_oh_flat[:, i].nonzero()]).sum(axis=0)
    return normalize(prob_xi_ci, norm='l1', axis=0)


def cal_log_likelihood(eps_graphs, cts_oh, cns_oh, prob_xi_ci, beta):
    log_likelihood = 0
    for sample in eps_graphs:
        for region, eps_graph in eps_graphs[sample].items():
            for i, neighbors in enumerate(eps_graph):
                prob_x0_ci = prob_xi_ci[cts_oh[sample][region][i].nonzero()[0][0]]
                prob_ci_cni = cal_prob_ci_cni(cns_oh[sample][region][neighbors], beta)
                max_ci = np.argmax(np.log(prob_x0_ci) + np.log(prob_ci_cni))
                log_likelihood += np.log(prob_x0_ci[max_ci]) + np.log(prob_ci_cni[max_ci]) - np.log((prob_x0_ci * prob_ci_cni).sum())
    return log_likelihood


def build_eps_graphs(ds, eps, max_neighbors):
    data = ds.data
    eps_graphs = ds.get_data_tpl_copy()
    for sample in data:
        for image in data[sample]:
            locs = data[sample][image].locs
            if max_neighbors == -1:
                for i, is_neighbor in enumerate(pairwise_distances(locs) <= eps):
                    neighbors = np.nonzero(is_neighbor)[0]
                    neighbors = neighbors[neighbors != i]
                    eps_graphs[sample][image].append(neighbors)
            else:
                pdists, indices = NearestNeighbors(n_neighbors=int(max_neighbors)).fit(locs).kneighbors(locs)
                for i, is_neighbor in enumerate(pdists <= eps):
                    neighbors = indices[i][np.nonzero(is_neighbor)[0]]
                    neighbors = neighbors[neighbors != i]
                    eps_graphs[sample][image].append(neighbors)
    return eps_graphs


class HMRF(Base):
    """A CN smoothing technique.

    Parameters
    ----------
    ds : Dataset
        Dataset object for the input dataset.
    n_cns : int
        Number of CNs.
    eps : float
        Pixel radius of neighborhoods.
    beta : float, default=9
        Weight of each neighbor in the same CN.
    max_neighbors : int, default=-1
        Maximum number of neighbors considered, -1 for all.
    max_iter : int, default=50
        Maximum number of iterations.
    max_iter_no_change : int, default=3
        Stop if the loss does not change for some iterations.
    verbose : bool, default=True
        Whether to print CN entropy and mean instance size.
    """

    def __init__(self, ds, n_cns, eps, beta=9, max_neighbors=-1, max_iter=50, max_iter_no_change=3, verbose=True):
        super().__init__(ds, n_cns, verbose)
        self.eps = eps
        self.beta = beta
        self.max_neighbors = max_neighbors
        self.max_iter = max_iter
        self.max_iter_no_change = max_iter_no_change
    
    def fit(self, cns):
        eps_graphs = build_eps_graphs(self.ds, self.eps, self.max_neighbors)
        cts_oh, cns_oh_curr = self.ds.get_cts_oh(), cns_sg_to_oh(cns, self.n_cns)
        cts_oh_flat, cns_oh_flat_curr = dic_to_flat(cts_oh), dic_to_flat(cns_oh_curr)

        n_iter_no_change = 1
        simi_old = 0
        for _ in range(self.max_iter):
            prob_xi_ci = cal_prob_xi_ci(cts_oh_flat, cns_oh_flat_curr)
            cns_oh_flat_next = []
            for sample in eps_graphs:
                for image in eps_graphs[sample]:
                    for i, neighbors in enumerate(eps_graphs[sample][image]):
                        prob_x0_ci = prob_xi_ci[cts_oh[sample][image][i].nonzero()[0][0]]
                        prob_ci_cni = cal_prob_ci_cni(cns_oh_curr[sample][image][neighbors], self.beta)
                        max_ci = np.argmax(np.log(prob_x0_ci) + np.log(prob_ci_cni))
                
                        cns_oh_tmp = np.zeros(prob_xi_ci.shape[1], dtype=int)
                        cns_oh_tmp[max_ci] = 1
                        cns_oh_flat_next.append(cns_oh_tmp)
            cns_oh_flat_next = np.array(cns_oh_flat_next)

            simi = (np.argmax(cns_oh_flat_next, axis=1) == np.argmax(cns_oh_flat_curr, axis=1)).mean()
                
            cns_oh_curr, cns_oh_flat_curr = self.ds.flat_to_dic(cns_oh_flat_next), cns_oh_flat_next
            n_iter_no_change = n_iter_no_change + 1 if simi == simi_old else 1
            simi_old = simi

            if n_iter_no_change == self.max_iter_no_change:
                break
        
        cns_smoothed = cns_oh_to_sg(cns_oh_curr)
        if self.verbose:
            cns_info(self.ds.data, self.n_cns, cns_smoothed)
        return cns_smoothed
