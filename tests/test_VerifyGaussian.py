# Set environment variables
import os
import glob

os.environ['pyanen_boot_repeats'] = '30'

# Load other modules
import pytest

import numpy as np
import matplotlib.pyplot as plt

from PyAnEn import VerifyProbGaussian


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
    'sigma': np.random.rand(*init_shape),
}


def test_cdf():
    
    verify = VerifyProbGaussian(f, o)
    probs = verify._cdf(below=10)
    assert np.all(probs==0.5)
    

def test_prob_to_ens():
    
    os.environ['pyanen_tqdm_workers'] = '4'
    
    verify = VerifyProbGaussian(f, o)
    
    with pytest.raises(AssertionError):
        ens = verify._prob_to_ens()
        
    verify = VerifyProbGaussian(f, o, n_sample_members=15, move_sampled_ens_axis=-2)
    ens = verify._prob_to_ens()
    
    l_shape = list(ens.shape)
    r_shape = list(o.shape)
    r_shape.insert(-1, 15)
    
    assert l_shape == r_shape, '{} == {}'.format(l_shape, r_shape)


def test_avg_axis():
    
    def _inner_(f, o, avg_axis, slice_idx, threshold, tag):
        verify1 = VerifyProbGaussian(f[0], o[0], avg_axis=avg_axis)
        verify2 = VerifyProbGaussian(f[1], o[1])
        verify3 = VerifyProbGaussian(f[2], o[2])
        
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
        
        assert np.all(r1[0][slice_idx, :] == r2[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
        assert np.all(r2[0] == r3[0]), 'Failed at spread skill {} with {}'.format(tag, avg_axis)
    
    # Evaluate testing routine
    init_shape = [5, 6, 7, 8]
    o = np.full(init_shape, 10)
    f = {
        'mu': np.full(init_shape, 10),
        'sigma': np.random.rand(*init_shape),
    }
    
    _inner_([f,
             {'mu': f['mu'][[3]], 'sigma': f['sigma'][[3]]},
             {'mu': f['mu'][3], 'sigma': f['sigma'][3]}],
            [o, o[[3]], o[3]], (1, 2, 3), 3, 17, 'first dimension')
    
    _inner_([f,
             {'mu': f['mu'][:, [3]], 'sigma': f['sigma'][:, [3]]},
             {'mu': f['mu'][:, 3], 'sigma': f['sigma'][:, 3]}],
            [o, o[:, [3]], o[:, 3]], (0, 2, 3), 3, 17, 'middle dimension')
    
    _inner_([f,
             {'mu': f['mu'][:, :, :, [3]], 'sigma': f['sigma'][:, :, :, [3]]},
             {'mu': f['mu'][:, :, :, 3], 'sigma': f['sigma'][:, :, :, 3]}],
            [o, o[:, :, :, [3]], o[:, :, :, 3]], (0, 1, 2), 3, 17, 'first dimension')

def test_saving():
    
    # The most basic use case without any saving
    verify = VerifyProbGaussian(f, o)
    verify.brier(below=10)
    
    # Specify the working directory
    verify = VerifyProbGaussian(f, o, working_directory='tmp')
    verify.brier(below=10)
    assert os.path.exists('tmp/brier_over_None_below_10.pkl')
    assert os.path.exists('tmp/cdf_over_None_below_10.pkl')
    os.remove('tmp/brier_over_None_below_10.pkl')
    os.remove('tmp/cdf_over_None_below_10.pkl')
    os.removedirs('tmp')
    
    verify = VerifyProbGaussian(f, o, working_directory='tmp')
    verify.brier(below=10, save_name='Happy  ')
    assert os.path.exists('tmp/Happy___over_None_below_10.pkl')
    assert os.path.exists('tmp/cdf_over_None_below_10.pkl')
    os.remove('tmp/Happy___over_None_below_10.pkl')
    os.remove('tmp/cdf_over_None_below_10.pkl')
    os.removedirs('tmp')
    
    verify = VerifyProbGaussian(f, o, working_directory='tmp')
    verify.brier(below=np.random.rand(*init_shape))
    assert len(glob.glob('tmp/*.pkl')) == 0
    os.removedirs('tmp')
    
    verify = VerifyProbGaussian(f, o, working_directory='tmp')
    verify.brier(below=np.random.rand(*init_shape), save_name='I want this')
    assert len(glob.glob('tmp/*.pkl')) == 0
    os.removedirs('tmp')
    
    verify = VerifyProbGaussian(f, o, working_directory='tmp')
    verify.brier(below=np.random.rand(*init_shape), save_name='LITERAL_I want this')
    assert len(glob.glob('tmp/*.pkl')) == 1
    assert os.path.exists('tmp/I_want_this.pkl')
    os.remove('tmp/I_want_this.pkl')
    os.removedirs('tmp')
    
