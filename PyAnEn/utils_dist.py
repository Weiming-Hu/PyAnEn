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
import os
import dill as pickle

import numpy as np

from scipy import stats
from distutils import util
from functools import partial
from tqdm.contrib.concurrent import process_map
from .dist_TruncatedGamma import truncgamma_gen

_LARGE_NUMBER_ = 10000


def _parallel_process(wrapper, iterable):
    
    cores = int(os.environ['pyanen_tqdm_workers'])
    chunksize = int(os.environ['pyanen_tqdm_chunksize'])
    parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
    assert cores > 1
    assert parallelize_axis < 0, 'parallelize_axis needs to be negative, counting from the end of dimensions, excluding the ensemble axis'
    
    ens = process_map(wrapper, iterable,
                      disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                      leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                      chunksize=chunksize, max_workers=cores)
    
    return np.concatenate(ens, axis=parallelize_axis)


def wrapper_truncgamma_rvs(x, n_sample_members): return truncgamma_gen()(a=(-x[2])/x[1], b=_LARGE_NUMBER_, s=x[0], scale=x[1], loc=x[2]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_truncgamma_cdf(x, t): return truncgamma_gen().cdf(x=t, a=(-x[2])/x[1], b=_LARGE_NUMBER_, s=x[0], scale=x[1], loc=x[2])
def wrapper_truncnorm_rvs(x, n_sample_members): return stats.truncnorm(a=-x[0]/x[1], b=_LARGE_NUMBER_, loc=x[0], scale=x[1]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_truncnorm_cdf(x, t): return stats.truncnorm.cdf(x=t, loc=x[0], scale=x[1], a=-x[0]/x[1], b=_LARGE_NUMBER_)
def wrapper_norm_rvs(x, n_sample_members): return stats.norm(loc=x[0], scale=x[1]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_norm_cdf(x, t): return stats.norm.cdf(x=t, loc=x[0], scale=x[1])

    
def sample_dist_gaussian(mu, sigma, n_sample_members=15, move_axis=-1, truncated=False):
    
    assert mu.shape == sigma.shape
    assert isinstance(move_axis, int)
    
    arr_shape = list(mu.shape)
    
    # Random samples
    cores = int(os.environ['pyanen_tqdm_workers'])
    
    if cores == 1:
        if truncated:
            # Truncate at zero
            # a is set to the clip value effectively at zero
            # b is set to a large number to mimic infinite 
            #
            ens = stats.truncnorm(a=-mu/sigma, b=_LARGE_NUMBER_, loc=mu, scale=sigma).rvs([n_sample_members] + arr_shape)
        else:
            ens = stats.norm(loc=mu, scale=sigma).rvs([n_sample_members] + arr_shape)
            
    else:
        parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
        iterable = zip(
            np.split(mu, mu.shape[parallelize_axis], parallelize_axis),
            np.split(sigma, mu.shape[parallelize_axis], parallelize_axis)
        )

        if truncated:
            wrapper = partial(wrapper_truncnorm_rvs, n_sample_members=n_sample_members)
        else:
            wrapper = partial(wrapper_norm_rvs, n_sample_members=n_sample_members)
        
        ens = _parallel_process(wrapper, iterable)
            
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
    cores = int(os.environ['pyanen_tqdm_workers'])
    
    if cores == 1:
        ens = truncgamma_gen()(a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, scale=scale, loc=shift).rvs([n_sample_members] + arr_shape)
        
    else:
        parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
        iterable = zip(
            np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
            np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
            np.split(shift, shift.shape[parallelize_axis], parallelize_axis),
        )
        
        wrapper = partial(wrapper_truncnorm_rvs, n_sample_members=n_sample_members)
        
        ens = _parallel_process(wrapper, iterable)
        
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def cdf_gaussian(mu, sigma, over=None, below=None, truncated=False):
    
    assert isinstance(truncated, bool)
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    cores = int(os.environ['pyanen_tqdm_workers'])        
    
    if below is None:
        
        if cores == 1:
            if truncated:
                probs = stats.truncnorm.cdf(x=over, loc=mu, scale=sigma, a=-mu/sigma, b=_LARGE_NUMBER_)
            else:
                probs = stats.norm.cdf(x=over, loc=mu, scale=sigma)
                
        else:
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
            iterable = zip(
                np.split(mu, mu.shape[parallelize_axis], parallelize_axis),
                np.split(sigma, sigma.shape[parallelize_axis], parallelize_axis),
            )
            
            if truncated:
                wrapper = partial(wrapper_truncnorm_cdf, t=over)
            else:
                wrapper = partial(wrapper_norm_cdf, t=over)
            
            probs = _parallel_process(wrapper, iterable)
            
        probs = 1 - probs
        
    else:
        
        if cores == 1:
            if truncated:
                probs = stats.truncnorm.cdf(x=below, loc=mu, scale=sigma, a=-mu/sigma, b=_LARGE_NUMBER_)
            else:
                probs = stats.norm.cdf(x=below, loc=mu, scale=sigma)
        
        else:
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
            iterable = zip(
                np.split(mu, mu.shape[parallelize_axis], parallelize_axis),
                np.split(sigma, sigma.shape[parallelize_axis], parallelize_axis),
            )
            
            if truncated:
                wrapper = partial(wrapper_truncnorm_cdf, t=below)
            else:
                wrapper = partial(wrapper_norm_cdf, t=below)
            
            probs = _parallel_process(wrapper, iterable)
            
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    
    return probs


def cdf_csgd(unshifted_mu, sigma, shift, over=None, below=None):
    
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    # Calculate distribution parameters
    shape = (unshifted_mu / sigma) ** 2
    scale = sigma ** 2 / unshifted_mu
    
    cores = int(os.environ['pyanen_tqdm_workers'])        
    
    if below is None:
        
        if cores == 1:
            probs = truncgamma_gen().cdf(x=over, a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, scale=scale, loc=shift)
        else:
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
            
            iterable = zip(
                np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
                np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
                np.split(shift, shift.shape[parallelize_axis], parallelize_axis),
            )
            
            wrapper = partial(wrapper_truncgamma_cdf, t=over)
            probs = _parallel_process(wrapper, iterable)
            
        probs = 1 - probs
        
    else:
        
        if cores == 1:
            probs = truncgamma_gen().cdf(x=below, a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, loc=shift, scale=scale)
        else:
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
            
            iterable = zip(
                np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
                np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
                np.split(shift, shift.shape[parallelize_axis], parallelize_axis),
            )
            
            wrapper = partial(wrapper_truncgamma_cdf, t=below)
            probs = _parallel_process(wrapper, iterable)
            
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    
    return probs
