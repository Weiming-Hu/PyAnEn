import numpy as np

from PyAnEn import VerifyProbGamma
from PyAnEn.utils_approximation import integrate


def test_mean():

    o = np.random.rand(3, 4, 5) + 10

    f = {
        'unshifted_mu': np.random.rand(3, 4, 5) + 10,
        'sigma': np.random.rand(3, 4, 5) + 1,
        'shift': np.full((3, 4, 5), 0),
    }

    f['mu'] = f['unshifted_mu']

    verifier = VerifyProbGamma(f=f, o=o)
    i_mu = integrate(verifier, 'mean', integration_range=(-100, 100), nbins=10000)
    assert np.abs(i_mu - f['mu']).sum() < 1e-5


def test_variance():

    o = np.random.rand(3, 4, 5) + 10

    f = {
        'unshifted_mu': np.random.rand(3, 4, 5) + 10,
        'sigma': np.random.rand(3, 4, 5) + 1,
        'shift': np.full((3, 4, 5), 0),
    }

    f['mu'] = f['unshifted_mu']

    verifier = VerifyProbGamma(f=f, o=o)
    i_var = integrate(verifier, 'variance', integration_range=(-100, 100), nbins=50000)
    
    assert np.abs(i_var - f['sigma'] ** 2).mean() < 1e-4

    
def test_crps():
    o = np.random.rand(3, 4, 5) + 10
    
    f = {
        'unshifted_mu': np.random.rand(3, 4, 5) + 10,
        'sigma': np.random.rand(3, 4, 5) + 1,
        'shift': np.full((3, 4, 5), 0),
    }

    f['mu'] = f['unshifted_mu']

    verifier = VerifyProbGamma(f=f, o=o)
    i_crps = integrate(verifier, 'brier', integration_range=(-100, 100), nbins=10000)
    assert i_crps.shape == (3, 4, 5)
    
    a_crps = verifier.set_avg_axis(None).crps()
    assert np.abs(a_crps - i_crps.mean()) < 1e-3
    