import numpy as np

from PyAnEn import VerifyProbGammaHurdle


def test_shape():
    
    o = np.random.rand(10, 4, 5) + 10
    
    f = {
        'mu': np.random.rand(10, 4, 5) + 10,
        'sigma': np.random.rand(10, 4, 5) + 1,
        'pop': np.random.rand(10, 4, 5),
    }

    verifier = VerifyProbGammaHurdle(f=f, o=o, avg_axis=1)

    crps = verifier.crps()
    assert crps.shape == (10, 5)

    crps = verifier.set_boot_samples(100).crps()
    assert crps.shape == (3, 10, 5)
    assert np.all(crps[0] < crps[1])
    assert np.all(crps[1] < crps[2])
    