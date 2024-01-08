from abc import ABC, abstractmethod

class Base(ABC):
    """Base class for CN identifiers."""

    def __init__(self, n_cns, exclude_cts, seed, verbose):
        self.n_cns = n_cns
        self.exclude_cts = exclude_cts
        self.seed = seed
        self.verbose = verbose
        self.feats = None
    
    @abstractmethod
    def fit(self, ds):
        """Idenfity CNs.

        Parameters
        ----------
        ds : Dataset
            Dataset object for the input dataset.
        """
