# Set environment variables
import os

os.environ['pyanen_boot_repeats'] = '30'

# Load other modules
import pytest

import numpy as np
import matplotlib.pyplot as plt

from PyAnEn import VerifyProbCSGD


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
    
    verify = VerifyProbCSGD(f, o)
    
    with pytest.raises(AssertionError):
        ens = verify._prob_to_ens()
        
    verify = VerifyProbCSGD(f, o, n_sample_members=15, move_sampled_ens_axis=-2)
    ens = verify._prob_to_ens()
    
    l_shape = list(ens.shape)
    r_shape = list(o.shape)
    r_shape.insert(-1, 15)
    
    assert l_shape == r_shape, '{} == {}'.format(l_shape, r_shape)
