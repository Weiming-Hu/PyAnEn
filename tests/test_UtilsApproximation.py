import os
import numpy as np

from PyAnEn import VerifyProbGamma
from PyAnEn import VerifyProbGaussian
from PyAnEn.utils_approximation import Integration


def test_mean():

    o = np.random.rand(3, 4, 5) + 10

    f = {
        'unshifted_mu': np.random.rand(3, 4, 5) + 10,
        'sigma': np.random.rand(3, 4, 5) + 1,
        'shift': np.full((3, 4, 5), 0),
    }

    f['mu'] = f['unshifted_mu']

    verifier = VerifyProbGamma(f=f, o=o)
    i_mu = Integration(verifier, integration_range=(-100, 100), nbins=10000).mean()
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
    i_var = Integration(verifier, integration_range=(-100, 100), nbins=50000).variance()
    
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
    i_crps = Integration(verifier, integration_range=(-100, 100), nbins=10000).crps()
    assert i_crps.shape == (3, 4, 5)
    
    a_crps = verifier.set_avg_axis(None).crps()
    assert np.abs(a_crps - i_crps.mean()) < 1e-3
    
    
def test_batch_variance():

    f = {
        'mu': np.random.rand(3, 4, 5) * 10,
        'sigma': np.abs(np.random.rand(3, 4, 5)),
    }

    o = f['mu']

    verifier = VerifyProbGaussian(f=f, o=o)
    f1 = verifier.f_determ()
    
    integrator = Integration(verifier=verifier, integration_range=(-100, 100), nbins=10000, less_memory=False)
    f2 = integrator.mean()

    integrator_less = Integration(verifier=verifier, integration_range=(-100, 100), nbins=10000, less_memory=True)
    f3 = integrator_less.mean()
    
    assert np.abs(f1 - f2).mean() < 1e-3
    assert np.abs(f2 - f3).mean() < 1e-2
    assert np.abs(f1 - f3).mean() < 1e-2

    
def test_parallel():
    os.environ['pyanen_tqdm_workers'] = '2'
    test_batch_variance()


def test_sum_mul():
    o = np.random.rand(10, 20, 30)
    ret1 = o.sum(axis=0)
    ret2 = Integration._memmap_sum_mul(o, np.full(o.shape, 1), o.dtype)
    assert np.abs(ret2 - ret1).mean() < 1e-5
