from abc import ABC, abstractmethod

class Base(ABC):
    """Base class for CN smoothers."""

    def __init__(self, ds, n_cns, verbose):
        self.ds = ds
        self.n_cns = n_cns
        self.verbose = verbose
    
    @abstractmethod
    def fit(self, cns):
        """Smooth CNs.

        Parameters
        ----------
        cns : dict
            Original CN results.
        """
