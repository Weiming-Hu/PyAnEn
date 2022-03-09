#############################################################
# Author: Weiming Hu <weiminghu@ucsd.edu>                   #
#                                                           #
#         Center for Western Weather and Water Extremes     #
#         Scripps Institution of Oceanography               #
#         UC San Diego                                      #
#                                                           #
#         https://weiming-hu.github.io/                     #
#         https://cw3e.ucsd.edu/                            #
#                                                           #
# Date of Creation: 2022/03/07                              #
#############################################################
#
# Utility functions for distributions
#

import numpy as np

from scipy import stats

    
def sample_dist_gaussian(mu, sigma, n_sample_members=15, move_axis=-1):
    
    assert mu.shape == sigma.shape
    assert isinstance(move_axis, int)
    
    arr_shape = list(mu.shape)
    
    # Random samples
    ens = stats.norm(loc=mu, scale=sigma).rvs([n_sample_members] + arr_shape)
    
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def sample_dist_csgd(unshifted_mu, sigma, shift, shift_sign=1, n_sample_members=15, move_axis=-1):
    
    assert shift.shape == sigma.shape
    assert shift.shape == unshifted_mu.shape
    assert shift_sign == 1 or shift_sign == -1
    assert isinstance(move_axis, int)
    
    arr_shape = list(unshifted_mu.shape)
    
    # Calculate distribution parameters
    shape = (unshifted_mu / sigma) ** 2
    scale = sigma ** 2 / unshifted_mu
    
    # Random samples
    ens = stats.gamma(a=shape, scale=scale, loc=shift * shift_sign).rvs([n_sample_members] + arr_shape)
    
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def cdf_gaussian(mu, sigma, over=None, below=None, truncated=False):
    
    assert isinstance(truncated, bool)
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if below is None:
        probs = stats.norm.cdf(x=over, loc=mu, scale=sigma)
        
        if truncated:
            probs[over < 0] = 0
            
        probs = 1 - probs
        
    else:
        probs = stats.norm.cdf(x=below, loc=mu, scale=sigma)
        
        if truncated:
            probs[below < 0] = 0
    
    return probs


def cdf_csgd(unshifted_mu, sigma, shift, shift_sign=1, over=None, below=None):
    
    assert (over is None) ^ (below is None), 'Must specify over or below'
    assert shift_sign == 1 or shift_sign == -1
    
    # Calculate distribution parameters
    shape = (unshifted_mu / sigma) ** 2
    scale = sigma ** 2 / unshifted_mu
    
    if below is None:
        probs = stats.gamma.cdf(x=over, a=shape, scale=scale, loc=shift * shift_sign)
        probs[over < 0] = 0
        probs = 1 - probs
    else:
        probs = stats.gamma.cdf(x=below, a=shape, scale=scale, loc=shift * shift_sign)
        probs[below < 0] = 0
    
    return probs
