# Set environment variables
import os

os.environ['pyanen_boot_repeats'] = '30'

# Load other modules
import pytest

import numpy as np

from PyAnEn import VerifyDeterm


# Set seed
np.random.seed(42)

# Calling these methods will raise exceptions
non_existing_methods = [
    'rank_hist',
    'reliability',
    'roc',
    'sharpness',
    'crps',
    'spread',
    'brier',
    'binned_spread_skill',
]

avg_axis_to_test = [
    None,
    0, 1, 2, 3,
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
    (0, 1, 2), (1, 2, 3),
    (0, 1, 2, 3)
]

# Create data for testing
init_shape = [11, 12, 13, 14]
o = np.random.randint(10, 100, size=init_shape)
f = o + o * np.random.rand()

def test_non_exsiting_methods():
    verify = VerifyDeterm(f, o)
    
    for m in non_existing_methods:
        func = getattr(verify, m)
        
        with pytest.raises(NotImplementedError):
            func()

def test_sq_error():
    for avg_axis in avg_axis_to_test:
        
        ###########
        # No boot #
        ###########
        
        verify = VerifyDeterm(f, o, avg_axis)
        
        l_hand_no_boot = verify.sq_error()
        r_hand = ((f - o) ** 2).mean(avg_axis)
        
        msg = 'Failed for avg_axis ({}) without boot'.format(avg_axis)
        assert np.all(l_hand_no_boot == r_hand), msg
        
        ########
        # Boot #
        ########
        
        verify = VerifyDeterm(f, o, avg_axis, boot_samples=100)
        
        l_hand_boot = verify.sq_error()
        
        msg = 'Failed for avg_axis ({}) with boot'.format(avg_axis)
        
        if avg_axis is None:
            assert l_hand_boot.shape[0] == 3 and len(l_hand_boot.shape) == 1, msg
        elif isinstance(avg_axis, int):
            assert list(l_hand_boot.shape) == [3] + [f.shape[i] for i in range(len(f.shape)) if i != avg_axis], msg
        else:
            assert list(l_hand_boot.shape) == [3] + [f.shape[i] for i in range(len(f.shape)) if i not in avg_axis], msg
        
        ############################
        # Compare boot and no boot #
        ############################
        
        assert np.all(l_hand_boot[0] < l_hand_no_boot) and np.all(l_hand_boot[2] > l_hand_no_boot)

def test_ab_error():
    verify = VerifyDeterm(f, o, working_directory=os.path.expanduser('~/github/PyAnEn/tmp'))
    assert verify.ab_error() == np.abs(f - o).mean()
    assert os.path.exists(os.path.expanduser('~/github/PyAnEn/tmp/ab_error.pkl'))
    assert verify.ab_error() == np.abs(f - o).mean()
    os.remove(os.path.expanduser('~/github/PyAnEn/tmp/ab_error.pkl'))
    os.rmdir(os.path.expanduser('~/github/PyAnEn/tmp'))


def test_avg_axis():
    
    # Define the testing routine
    def _inner_(f, o, avg_axis, slice_idx, tag):
        verify1 = VerifyDeterm(f[0], o[0], avg_axis=avg_axis)
        verify2 = VerifyDeterm(f[1], o[1])
        verify3 = VerifyDeterm(f[2], o[2])

        r1 = verify1.rmse()
        r2 = verify2.rmse()
        r3 = verify3.rmse()

        assert np.abs(r1[slice_idx] - r2) < 1e-5, 'Failed at {} with {}'.format(tag, avg_axis)
        assert np.abs(r2 - r3) < 1e-5, 'Failed at {} with {}'.format(tag, avg_axis)
    
    # Evaluate testing routine
    o = np.random.randint(10, 100, size=init_shape)
    f = np.random.normal(o)
    _inner_([f, f[[3]], f[3]], [o, o[[3]], o[3]], (1, 2, 3), 3, 'first dimension')
    _inner_([f, f[:, [3]], f[:, 3]], [o, o[:, [3]], o[:, 3]], (0, 2, 3), 3, 'middle dimension')
    _inner_([f, f[:, :, :, [3]], f[:, :, :, 3]], [o, o[:, :, :, [3]], o[:, :, :, 3]], (0, 1, 2), 3, 'last dimension')
    