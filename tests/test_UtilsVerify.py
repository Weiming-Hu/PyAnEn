import os

import numpy as np
import matplotlib.pyplot as plt

from ranky import rankz
from PyAnEn.utils_ss import binned_spread_skill
from PyAnEn.utils_verify import iou_determ
from PyAnEn.utils_verify import ens_to_prob
from PyAnEn.utils_verify import rank_histogram
from PyAnEn.utils_verify import reliability_diagram


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


def test_EnsToProb_KDE_arr():
    os.environ['pyanen_ens_to_prob_method'] = 'kde'
    
    over = np.random.normal(10, 2, (3, 4, 10))
    f = np.random.normal(10, 2, (3, 4, 10, 50))
    
    os.environ['pyanen_tqdm_workers'] = '1'
    f_serial = ens_to_prob(f, 3, over=over)
    
    os.environ['pyanen_tqdm_workers'] = '2'
    f_parallel = ens_to_prob(f, 3, over=over)
    
    assert np.all(f_serial == f_parallel)
    assert f_serial.shape == (3, 4, 10)
    assert f_parallel.shape == (3, 4, 10)


def test_EnsToProb_Moments_arr():
    os.environ['pyanen_ens_to_prob_method'] = 'moments'
    
    over = np.random.normal(10, 2, (3, 4, 10))
    f = np.random.normal(10, 2, (3, 4, 10, 50))
    
    os.environ['pyanen_tqdm_workers'] = '1'
    f_serial = ens_to_prob(f, 3, over=over)
    
    os.environ['pyanen_tqdm_workers'] = '2'
    f_parallel = ens_to_prob(f, 3, over=over)
    
    assert np.all(f_serial == f_parallel)
    assert f_serial.shape == (3, 4, 10)
    assert f_parallel.shape == (3, 4, 10)
    
    
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
    
    
def test_Reliability():

    # Initialization
    f_prob = np.random.rand(4, 1000, 2, 3)
    o_binary = np.random.rand(4, 1000, 2, 3)
    nbins = 15
    
    ###########
    # No Boot #
    ###########

    # Global
    ret_global = reliability_diagram(f_prob, o_binary, nbins, [1])
    
    assert len(ret_global) == 4
    assert ret_global[0].shape[-1] == nbins
    assert ret_global[1].shape[-1] == nbins

    for i in range(4):
        for j in range(2):
            for k in range(3):
                ret_slice = reliability_diagram(f_prob[i, :, j, k], o_binary[i, :, j, k], nbins, None)
                
                assert np.all(ret_global[0][i, j, k] == ret_slice[0])
                assert np.all(ret_global[1][i, j, k] == ret_slice[1])
                assert np.all(ret_global[2][i, j, k] == ret_slice[2].to_numpy())
            
    ######## 
    # Boot #
    ######## 
    
    # Only testing dimensions   
    ret_global = reliability_diagram(f_prob, o_binary, nbins, [1], boot_samples=50)
    
    assert len(ret_global) == 4
    assert ret_global[0].shape[-1] == 3
    assert ret_global[1].shape[-1] == 3
    assert ret_global[0].shape[-2] == nbins
    assert ret_global[1].shape[-2] == nbins
                
                
def test_BinnedSpreadSkill():

    # Initialization
    variance = np.random.rand(4, 1000, 2, 3)
    sq_error = np.random.rand(4, 1000, 2, 3)
    nbins = 15

    # Global
    ret_global = binned_spread_skill(variance, sq_error, nbins, [1])
    
    assert len(ret_global) == 2
    assert ret_global[0].shape[-1] == nbins
    assert ret_global[1].shape[-1] == nbins

    for i in range(4):
        for j in range(2):
            for k in range(3):
                ret_slice = binned_spread_skill(variance[i, :, j, k], sq_error[i, :, j, k], nbins, None)
                
                assert np.all(ret_global[0][i, j, k] == ret_slice[0])
                assert np.all(ret_global[1][i, j, k] == ret_slice[1])
            
    ######## 
    # Boot #
    ######## 
    
    # Only testing dimensions   
    ret_global = reliability_diagram(variance, sq_error, nbins, [1], boot_samples=50)
    
    assert len(ret_global) == 4
    assert ret_global[0].shape[-1] == 3
    assert ret_global[1].shape[-1] == 3
    assert ret_global[0].shape[-2] == nbins
    assert ret_global[1].shape[-2] == nbins
    
