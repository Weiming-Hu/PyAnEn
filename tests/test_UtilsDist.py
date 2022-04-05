import os

import numpy as np
import matplotlib.pyplot as plt

from ranky import rankz
from PyAnEn.utils_dist import cdf_gaussian, cdf_gamma_hurdle


def test_gaussian():
    
    init_shape = [10, 20]
    mu = np.random.rand(*init_shape) + 1
    sigma = np.random.rand(*init_shape) + 0.01
    below = 1
    
    os.environ['pyanen_tqdm_workers'] = '1'
    r1 = cdf_gaussian(mu, sigma, below=below)
    os.environ['pyanen_tqdm_workers'] = '4'
    r2 = cdf_gaussian(mu, sigma, below=below)
    
    assert np.all(r1 == r2)


def test_gamma_hurdle():
    
    os.environ['pyanen_tqdm_workers'] = '1'
    
    pop = np.array([[0, 0.4], [0.7, 1]])
    mu = np.array([[1, 2], [3, 4]])
    sigma = np.array([[1, 2 ], [3, 4]])
    y = np.array([[0, 0], [3, 4]])
    
    
    r1 = cdf_gamma_hurdle(pop, mu, sigma, below=y)
    r2 = cdf_gamma_hurdle(pop, mu, sigma, below=0)
    r3 = cdf_gamma_hurdle(pop, mu, sigma, below=3)
    
    assert r1.shape == r2.shape == pop.shape
    assert r1[0, 0] == r2[0, 0]
    assert r1[0, 1] == r2[0, 1]
    assert r1[1, 0] == r3[1, 0]
    
    r1 = cdf_gamma_hurdle(pop, mu, sigma, below=y)
    r2 = cdf_gamma_hurdle(pop, mu, sigma, below=0)
    r3 = cdf_gamma_hurdle(pop, mu, sigma, below=3)
    
    assert r1.shape == r2.shape == pop.shape
    assert r1[0, 0] == r2[0, 0]
    assert r1[0, 1] == r2[0, 1]
    assert r1[1, 0] == r3[1, 0]
    
    os.environ['pyanen_tqdm_workers'] = '4'
    p1 = cdf_gamma_hurdle(pop, mu, sigma, below=y)
    p2 = cdf_gamma_hurdle(pop, mu, sigma, below=0)
    p3 = cdf_gamma_hurdle(pop, mu, sigma, below=3)
    
    assert np.all(r1 == p1)
    assert np.all(r2 == p2)
    assert np.all(r3 == p3)
    