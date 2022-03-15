import os

import numpy as np
import matplotlib.pyplot as plt

from ranky import rankz
from PyAnEn.utils_dist import cdf_gaussian


def test_gaussian():
    
    init_shape = [10, 20]
    mu = np.random.rand(*init_shape) + 1
    sigma = np.random.rand(*init_shape) + 0.01
    over = 1
    
    os.environ['pyanen_tqdm_workers'] = '1'
    r1 = cdf_gaussian(mu, sigma, over=over)
    os.environ['pyanen_tqdm_workers'] = '4'
    r2 = cdf_gaussian(mu, sigma, over=over)
    
    assert np.all(r1 == r2)
    