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
    f_over = verify.cdf(over=18)
    f_below = verify.cdf(below=18)
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
        
        assert spread2.shape[-1] == skill2.shape[-1] == 3
        assert spread1.shape == spread2.shape[:-1]
        assert skill1.shape == skill2.shape[:-1]


def test_brier():
    verify = VerifyEnsemble(f, o, working_directory=os.path.expanduser('~/github/PyAnEn/tmp'))
    br = verify.brier(over=16.5)
    assert os.path.exists('./tmp/brier_over_16.5_below_None.pkl')
    assert os.path.exists('./tmp/cdf_over_16.5_below_None.pkl')
    assert os.path.exists('./tmp/cdf_over_None_below_16.5.pkl')
    br = verify.brier(over=16.5)
    os.remove('./tmp/brier_over_16.5_below_None.pkl')
    os.remove('./tmp/cdf_over_16.5_below_None.pkl')
    os.remove('./tmp/cdf_over_None_below_16.5.pkl')
    os.rmdir('./tmp')


def test_avg_axis():
    
    def _inner_(f, o, avg_axis, slice_idx, threshold, tag):
        verify1 = VerifyEnsemble(f[0], o[0], avg_axis=avg_axis)
        verify2 = VerifyEnsemble(f[1], o[1])
        verify3 = VerifyEnsemble(f[2], o[2])
        
        # CRPS

        r1 = verify1.crps()
        r2 = verify2.crps()
        r3 = verify3.crps()
        
        assert np.abs(r1[slice_idx] - r2) < 1e-5, 'Failed at CRPS {} with {}'.format(tag, avg_axis)
        assert np.abs(r2 - r3) < 1e-5, 'Failed at CRPS {} with {}'.format(tag, avg_axis)
        
        # Rank histogram
        r1 = verify1.rank_hist()
        r2 = verify2.rank_hist()
        r3 = verify3.rank_hist()
    
        if tag[:5] == 'first':
            assert np.all(r1[slice_idx] == r2.squeeze()), 'Failed at rank histogram {} with {}'.format(tag, avg_axis)
        elif tag[:6] == 'middle':
            assert np.all(r1[:, slice_idx] == r2.squeeze()), 'Failed at rank histogram {} with {}'.format(tag, avg_axis)
        elif tag[:4] == 'last':
            assert np.all(r1[:, :, :, slice_idx] == r2.squeeze()), 'Failed at rank histogram {} with {}'.format(tag, avg_axis)
        else:
            raise RuntimeError
        
        assert np.all(r2.squeeze()== r3), 'Failed at rank histogram {} with {}'.format(tag, avg_axis)
        
        # Reliability
        
        r2 = verify2.reliability(over=threshold)
        r3 = verify3.reliability(over=threshold)
        assert np.all(r2[0] == r3[0]), 'Failed at reliability {} with {}'.format(tag, avg_axis)
        
        # Spread skill
        r1 = verify1.binned_spread_skill()
        r2 = verify2.binned_spread_skill()
        r3 = verify3.binned_spread_skill()
        
        assert np.all(r1[0][slice_idx, :] == r2[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
        assert np.all(r2[0] == r3[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
    
    # Evaluate testing routine
    init_shape = [8, 9, 10, 11]
    o = np.random.randint(15, 20, size=init_shape)
    f = np.random.normal(o, 3, size=[30] + list(o.shape))
    _inner_([f, f[:, [3]], f[:, 3]], [o, o[[3]], o[3]], (1, 2, 3), 3, 17, 'first dimension')
    _inner_([f, f[:, :, [3]], f[:, :, 3]], [o, o[:, [3]], o[:, 3]], (0, 2, 3), 3, 17, 'middle dimension')
    _inner_([f, f[:, :, :, :, [3]], f[:, :, :, :, 3]], [o, o[:, :, :, [3]], o[:, :, :, 3]], (0, 1, 2), 3, 17, 'last dimension')
    
