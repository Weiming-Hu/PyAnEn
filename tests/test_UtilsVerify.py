import os

import numpy as np
import matplotlib.pyplot as plt

from ranky import rankz
from PyAnEn.utils_verify import iou_determ
from PyAnEn.utils_verify import ens_to_prob
from PyAnEn.utils_verify import rank_histogram


def test_RankHist():
    
    # This is the package I'm verifying against
    # https://github.com/oliverangelil/rankhistogram
    #
    
    obs = np.random.randint(5, 95, (10, 20, 30)) + 0.1
    ensemble = np.random.randint(1, 100, (40, 10, 20, 30))

    # Feed into rankz function
    mask = np.full((10, 20, 30), 1)
    theirs = rankz(obs, ensemble, mask)
    
    for ens_axis in [0, 1, 2, 3]:
        
        # Calculate my rank histograms
        ranks = rank_histogram(np.moveaxis(ensemble, 0, ens_axis), obs, ens_axis)
        mine = plt.hist(ranks.ravel(), bins=theirs[1])

        assert np.all(mine[0] == theirs[0]), 'Failed with ensemble axis {}'.format(ens_axis)


def test_EnsToProb_KDE():
    os.environ['pyanen_ens_to_prob_method'] = 'kde'
    
    f = np.random.normal(10, 2, (3, 4, 10, 50))
    
    os.environ['pyanen_tqdm_workers'] = '1'
    f_serial = ens_to_prob(f, 3, over=11)
    
    os.environ['pyanen_tqdm_workers'] = '2'
    f_parallel = ens_to_prob(f, 3, over=11)
    
    assert np.all(f_serial == f_parallel)
    assert f_serial.shape == (3, 4, 10)
    assert f_parallel.shape == (3, 4, 10)


def test_EnsToProb_Moments():
    os.environ['pyanen_ens_to_prob_method'] = 'moments'
    
    f = np.random.normal(10, 2, (3, 4, 10, 50))
    
    os.environ['pyanen_tqdm_workers'] = '1'
    f_serial = ens_to_prob(f, 3, over=11)
    
    os.environ['pyanen_tqdm_workers'] = '2'
    f_parallel = ens_to_prob(f, 3, over=11)
    
    assert np.all(f_serial == f_parallel)
    assert f_serial.shape == (3, 4, 10)
    assert f_parallel.shape == (3, 4, 10)
    
    
def test_DetermIOU():
    f = np.array([
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        [4, 5, 6.5, 7, 8, 7, 6, 5, 4],
        [2, 3, 4, 5, 6, 5, 4, 3, 2],
    ])
    
    o = np.array([
        [1, 2, 3, 4, 5, 6, 7, 6, 5],
        [4, 5, 6, 7, 8, 9, 10, 9, 8],
        [2, 3, 4, 5, 6, 7, 8, 7, 6],
    ])
    
    assert iou_determ(f=f, o=o, over=6) == 3/11
    
    o = np.array([
        [1, 2, 3, 4, 5, 6, 7, 6, 5],
        [4, 5, 6, 7, 8, 9, 10, 9, 8],
        [2, 3, 4, 5, 6, 7, 8, np.nan, 6],
    ])
    
    assert iou_determ(f=f, o=o, over=6) == 3/10
    
    f = np.array([
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        [4, 5, np.nan, 7, np.nan, 7, 6, 5, 4],
        [2, 3, 4, 5, 6, 5, 4, 3, 2],
    ])
    
    assert iou_determ(f=f, o=o, over=6) == 2/9
    