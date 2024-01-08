from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([Extension('_k_means_lloyd_reg', ['cntools/utils/_k_means_lloyd_reg.pyx'], include_dirs=[np.get_include()])])
)