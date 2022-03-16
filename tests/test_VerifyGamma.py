# Set environment variables
import os
from cffi import VerificationError

os.environ['pyanen_boot_repeats'] = '30'
os.environ['pyanen_lbeta_tensorflow'] = 'True'

# Load other modules
import pytest

import numpy as np
import matplotlib.pyplot as plt

from PyAnEn import VerifyProbGamma


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
init_shape = [5, 6, 7, 3]
o = np.full(init_shape, 10)
f = {
    'mu': np.full(init_shape, 10),
    'unshifted_mu': np.full(init_shape, 10),
    'shift': np.full(init_shape, 0),
    'sigma': np.random.randint(1, 2, size=init_shape),
}


def test_prob_to_ens():
    
    verify = VerifyProbGamma(f, o)
    
    with pytest.raises(AssertionError):
        ens = verify._prob_to_ens()
    
    os.environ['pyanen_tqdm_workers'] = '4'
    verify = VerifyProbGamma(f, o, n_sample_members=15, move_sampled_ens_axis=-2)
    ens = verify._prob_to_ens()
    
    l_shape = list(ens.shape)
    r_shape = list(o.shape)
    r_shape.insert(-1, 15)
    
    assert l_shape == r_shape, '{} == {}'.format(l_shape, r_shape)


def test_avg_axis():
    
    def _inner_(f, o, avg_axis, slice_idx, threshold, tag):
        verify1 = VerifyProbGamma(f[0], o[0], avg_axis=avg_axis)
        verify2 = VerifyProbGamma(f[1], o[1])
        verify3 = VerifyProbGamma(f[2], o[2])
        
        # CRPS

        r1 = verify1.crps()
        r2 = verify2.crps()
        r3 = verify3.crps()
        
        assert np.abs(r1[slice_idx] - r2) < 1e-5, 'Failed at CRPS {} with {}'.format(tag, avg_axis)
        assert np.abs(r2 - r3) < 1e-5, 'Failed at CRPS {} with {}'.format(tag, avg_axis)
        
        # Reliability
        
        r2 = verify2.reliability(over=threshold, nbins=10)
        r3 = verify3.reliability(over=threshold, nbins=10)
        assert np.all(r2[0] == r3[0]), 'Failed at reliability {} with {}'.format(tag, avg_axis)
        
        # Spread skill
        r1 = verify1.binned_spread_skill(nbins=10)
        r2 = verify2.binned_spread_skill(nbins=10)
        r3 = verify3.binned_spread_skill(nbins=10)
        
        assert np.all(r1[0][:, slice_idx] == r2[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
        assert np.all(r2[0] == r3[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
    
    # Evaluate testing routine
    init_shape = [5, 6, 7, 8]
    o = np.full(init_shape, 10)
    f = {
        'mu': np.full(init_shape, 10),
        'unshifted_mu': np.full(init_shape, 9),
        'shift': -np.full(init_shape, 1),
        'sigma': np.random.rand(*init_shape) + 1,
    }
    
    _inner_([f,
             {'mu': f['mu'][[3]], 'sigma': f['sigma'][[3]], 'unshifted_mu': f['unshifted_mu'][[3]], 'shift': f['shift'][[3]]},
             {'mu': f['mu'][3], 'sigma': f['sigma'][3], 'unshifted_mu': f['unshifted_mu'][3], 'shift': f['shift'][3]}],
            [o, o[[3]], o[3]], (1, 2, 3), 3, 17, 'first dimension')
    
    _inner_([f,
             {'mu': f['mu'][:, [3]], 'sigma': f['sigma'][:, [3]], 'unshifted_mu': f['unshifted_mu'][:, [3]], 'shift': f['shift'][:, [3]]},
             {'mu': f['mu'][:, 3], 'sigma': f['sigma'][:, 3], 'unshifted_mu': f['unshifted_mu'][:, 3], 'shift': f['shift'][:, 3]}],
            [o, o[:, [3]], o[:, 3]], (0, 2, 3), 3, 17, 'middle dimension')
    
    _inner_([f,
             {'mu': f['mu'][:, :, :, [3]], 'sigma': f['sigma'][:, :, :, [3]], 'unshifted_mu': f['unshifted_mu'][:, :, :, [3]], 'shift': f['shift'][:, :, :, [3]]},
             {'mu': f['mu'][:, :, :, 3], 'sigma': f['sigma'][:, :, :, 3], 'unshifted_mu': f['unshifted_mu'][:, :, :, 3], 'shift': f['shift'][:, :, :, 3]}],
            [o, o[:, :, :, [3]], o[:, :, :, 3]], (0, 1, 2), 3, 17, 'first dimension')


def test_reliability():
    
    init_shape = [5, 6, 7, 3]
    o = np.random.rand(*init_shape)
    f = {
        'mu': np.random.rand(*init_shape),
        'unshifted_mu': np.random.rand(*init_shape),
        'shift': -np.random.rand(*init_shape),
        'sigma': np.random.rand(*init_shape) * 10 + 0.01,
    }
    
    verifier_no_boot = VerifyProbGamma(f=f, o=o)
    verifier_boot = VerifyProbGamma(f=f, o=o, boot_samples=10000)
    
    y_pred1, y_true1, counts1 = verifier_no_boot.reliability(over=1)
    y_pred2, y_true2, counts2 = verifier_boot.reliability(over=1)
    
    mask = np.isfinite(y_pred1)
    y_pred1, y_pred2 = y_pred1[mask], y_pred2[mask, :]
    
    mask = np.isfinite(y_true1)
    y_true1, y_true2 = y_true1[mask], y_true2[mask, :]
    
    assert len(y_pred1.shape) == len(y_true1.shape) == 1
    assert len(y_pred2.shape) == len(y_pred2.shape) == 2
    assert np.all(y_pred2[:, 0] <= y_pred1)
    assert np.all(y_pred1 <= y_pred2[:, 2])
    assert np.all(y_true2[:, 0] <= y_true1)
    assert np.all(y_true1 <= y_true2[:, 2])
    assert np.all(counts1 == counts2)
    