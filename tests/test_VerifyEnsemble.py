# Set environment variables
import os

os.environ['pyanen_boot_repeats'] = '30'

# Load other modules
import pytest

import numpy as np
import matplotlib.pyplot as plt

from PyAnEn import VerifyEnsemble


# Set seed
np.random.seed(42)

avg_axis_to_test = [
    None,
    0, 1, 2, 3,
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
    (0, 1, 2), (1, 2, 3),
    (0, 1, 2, 3)
]

# Create data for testing
init_shape = [8, 9, 10, 11]
o = np.random.randint(15, 20, size=init_shape)
f = np.random.normal(o, 3, size=[30] + list(o.shape))


def test_rank_histogram_under_dispersive():
    
    o = np.random.normal(10, 8, size=init_shape)
    f = np.random.normal(10, 4, size=[30] + list(o.shape))
    f = np.moveaxis(f, 0, -1)
    verify = VerifyEnsemble(f, o)
    assert verify.ensemble_axis == -1
    rh = verify.rank_hist()
    assert rh.shape == o.shape
    ranks = plt.hist(rh.ravel(), bins=15)
    ranks[0][0] > ranks[0][7] < ranks[0][-1]
    
    
def test_rank_histogram_over_dispersive():
    
    o = np.random.normal(10, 2, size=init_shape)
    f = np.random.normal(10, 4, size=[30] + list(o.shape))
    f = np.moveaxis(f, 0, -1)
    verify = VerifyEnsemble(f, o)
    assert verify.ensemble_axis == -1
    rh = verify.rank_hist()
    assert rh.shape == o.shape
    ranks = plt.hist(rh.ravel(), bins=15)
    ranks[0][0] < ranks[0][7] > ranks[0][-1]


def test_ens_to_prob():
    
    verify = VerifyEnsemble(f, o)
    assert verify.ensemble_axis == 0
    f_over = verify._ens_to_prob(over=18)
    f_below = verify._ens_to_prob(below=18)
    assert np.count_nonzero(np.abs(f_over + f_below - 1) < 1e-2) > 0.9


def test_spread_skill():
    for axis in avg_axis_to_test:
        verify1 = VerifyEnsemble(f, o, avg_axis=axis)
        verify2 = VerifyEnsemble(f, o, avg_axis=axis, boot_samples=100)
        
        if axis is None:
            n_samples = np.prod(o.shape)
        elif isinstance(axis, int):
            n_samples = o.shape[axis]
        else:
            n_samples = np.prod([o.shape[i] for i in axis])
            
        if n_samples > 40:
            spread1, skill1 = verify1.binned_spread_skill(nbins=4)
            spread2, skill2 = verify2.binned_spread_skill(nbins=4)
        else:
            with pytest.warns(UserWarning):
                spread1, skill1 = verify1.binned_spread_skill(nbins=4)
                spread2, skill2 = verify2.binned_spread_skill(nbins=4)
        
        assert spread2.shape[0] == skill2.shape[0] == 3
        assert spread1.shape == spread2.shape[1:]
        assert skill1.shape == skill2.shape[1:]


def test_brier():
    verify = VerifyEnsemble(f, o, working_directory='./tmp')
    br = verify.brier(over=16.5)
    assert os.path.exists('./tmp/brier_16.5_None.npy')
    assert os.path.exists('./tmp/ens_to_prob_16.5_None.npy')
    br = verify.brier(over=16.5)
    os.remove('./tmp/ens_to_prob_16.5_None.npy')
    os.remove('./tmp/brier_16.5_None.npy')
    os.rmdir('./tmp')
    