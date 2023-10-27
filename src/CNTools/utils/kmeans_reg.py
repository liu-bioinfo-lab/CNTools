import numpy as np
from sklearn.cluster import KMeans
from functools import partial
import warnings
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.cluster._kmeans import _kmeans_single_elkan
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from sklearn.cluster._k_means_fast import _inertia_dense, _inertia_sparse
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_sparse
import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()})
from ._k_means_lloyd_reg import lloyd_iter_chunked_dense_reg


def _kmeans_single_lloyd_reg(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1, lam=0, edges=None):
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = partial(lloyd_iter_chunked_dense_reg, lam=lam, edges_indptr=edges.indptr if edges is not None else None, edges_indices=edges.indices if edges is not None else None)
        _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads)

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(f"Converged at iteration {i}: center shift "
                              f"{center_shift_tot} within tolerance {tol}.")
                    break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers,
                       weight_in_clusters, labels, center_shift, n_threads,
                       update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels)

    return labels, inertia, centers, i + 1


class KMeans_reg(KMeans):
    def __init__(self, n_clusters=8, *, init="k-means++", n_init=10, max_iter=300, tol=1e-4, 
                 precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True,
                 n_jobs='deprecated', algorithm='full', lam=0, edges=None) -> None:
        super().__init__(n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances=precompute_distances, verbose=verbose, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
        self.lam = lam
        self.edges = edges

    def fit(self, X, y=None, sample_weight=None):
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = partial(_kmeans_single_lloyd_reg, lam=self.lam, edges=self.edges)
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            kmeans_single = _kmeans_single_elkan

        best_inertia = None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=random_state)
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, centers_init, max_iter=self.max_iter,
                verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, n_threads=self._n_threads)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
