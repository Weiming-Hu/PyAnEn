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
from .dist_TruncatedGamma import truncgamma_gen

_LARGE_NUMBER_ = 10000

    
def sample_dist_gaussian(mu, sigma, n_sample_members=15, move_axis=-1, truncated=False):
    
    assert mu.shape == sigma.shape
    assert isinstance(move_axis, int)
    
    arr_shape = list(mu.shape)
    
    # Random samples
    if truncated:
        # Truncate at zero
        # a is set to the clip value effectively at zero
        # b is set to a large number to mimic infinite 
        #
        ens = stats.truncnorm(a=-mu/sigma, b=_LARGE_NUMBER_, loc=mu, scale=sigma).rvs([n_sample_members] + arr_shape)
    else:
        ens = stats.norm(loc=mu, scale=sigma).rvs([n_sample_members] + arr_shape)
    
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def sample_dist_csgd(unshifted_mu, sigma, shift, n_sample_members=15, move_axis=-1):
    
    assert shift.shape == sigma.shape
    assert shift.shape == unshifted_mu.shape
    assert isinstance(move_axis, int)
    
    arr_shape = list(unshifted_mu.shape)
    
    # Calculate distribution parameters
    shape = (unshifted_mu / sigma) ** 2
    scale = sigma ** 2 / unshifted_mu
    
    # Random samples
    ens = truncgamma_gen()(a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, scale=scale, loc=shift).rvs([n_sample_members] + arr_shape)
    
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def cdf_gaussian(mu, sigma, over=None, below=None, truncated=False):
    
    assert isinstance(truncated, bool)
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if below is None:
        if truncated:
            probs = stats.truncnorm.cdf(x=over, loc=mu, scale=sigma, a=-mu/sigma, b=_LARGE_NUMBER_)
        else:
            probs = stats.norm.cdf(x=over, loc=mu, scale=sigma)
            
        probs = 1 - probs
        
    else:
        if truncated:
            probs = stats.truncnorm.cdf(x=below, loc=mu, scale=sigma, a=-mu/sigma, b=_LARGE_NUMBER_)
        else:
            probs = stats.norm.cdf(x=below, loc=mu, scale=sigma)
    
    return probs


def cdf_csgd(unshifted_mu, sigma, shift, over=None, below=None):
    
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    # Calculate distribution parameters
    shape = (unshifted_mu / sigma) ** 2
    scale = sigma ** 2 / unshifted_mu
    
    if below is None:
        probs = truncgamma_gen().cdf(x=over, a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, loc=shift, scale=scale)
        probs = 1 - probs
    else:
        probs = truncgamma_gen().cdf(x=below, a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, loc=shift, scale=scale)
    return probs
