import os
import numpy as np

from PyAnEn import VerifyProbGamma
from PyAnEn import VerifyProbGaussian
from PyAnEn.utils_approximation import Integration


# os.environ['pyanen_tqdm_disable'] = 'False'
# os.environ['pyanen_tqdm_leave'] = 'True'


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
    
    print('Calculating f_determ ...')
    verifier = VerifyProbGaussian(f=f, o=o)
    f1 = verifier.f_determ()
    
    print('Integrate ...')
    integrator = Integration(verifier=verifier, integration_range=(-100, 100), nbins=10000, less_memory=False)
    f2 = integrator.mean()

    print('Integrate (less memory) ...')
    integrator_less = Integration(verifier=verifier, integration_range=(-100, 100), nbins=10000, less_memory=True)
    f3 = integrator_less.mean()
    
    # print('Saving ...')
    # np.save('tmp.pkl', np.stack([f1, f2, f3], axis=0))
    
    assert np.abs(f1 - f2).max() < 1e-2, 'Largest difference is {}'.format(np.abs(f1 - f2).max())
    assert np.abs(f2 - f3).max() < 1e-2, 'Largest difference is {}'.format(np.abs(f2 - f3).max())
    assert np.abs(f1 - f3).max() < 1e-2, 'Largest difference is {}'.format(np.abs(f1 - f3).max())


def test_parallel():
    old_val = os.environ['pyanen_tqdm_workers']
    os.environ['pyanen_tqdm_workers'] = '4'
    test_batch_variance()
    os.environ['pyanen_tqdm_workers'] = old_val


if __name__ == '__main__':
    test_parallel()
