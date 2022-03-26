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

import numpy as np

from scipy import stats
from distutils import util
from functools import partial
from .dist_truncgamma import truncgamma_gen
from tqdm.contrib.concurrent import process_map

_LARGE_NUMBER_ = 10000


def _parallel_process(wrapper, iterable, total, name=''):
    
    cores = int(os.environ['pyanen_tqdm_workers'])
    chunksize = int(os.environ['pyanen_tqdm_chunksize'])
    parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
    assert cores > 1
    assert parallelize_axis < 0, 'parallelize_axis needs to be negative, counting from the end of dimensions, excluding the ensemble axis'
    
    ens = process_map(wrapper, iterable, total=total,
                      disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                      leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                      chunksize=chunksize, max_workers=cores, desc='Process {}'.format(name))
    
    return np.concatenate(ens, axis=parallelize_axis)


def wrapper_gamma_rvs(x, n_sample_members):
    return stats.gamma(a=x[0], scale=x[1], loc=x[2]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_gamma_cdf(x):
    return stats.gamma.cdf(x=x[3], a=x[0], scale=x[1], loc=x[2])

def wrapper_truncgamma_rvs(x, n_sample_members):
    return truncgamma_gen()(a=(-x[2])/x[1], b=_LARGE_NUMBER_, s=x[0], scale=x[1], loc=x[2]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_truncgamma_cdf(x):
    return truncgamma_gen().cdf(x=[3], a=(-x[2])/x[1], b=_LARGE_NUMBER_, s=x[0], scale=x[1], loc=x[2])

def wrapper_gamma_hurdle_cdf(x):
    prob = 1. - x[0]
    mask = x[3] > 0.
    prob[mask] += x[0][mask] * stats.gamma.cdf(x=x[3][mask], a=x[1][mask], scale=x[2][mask])
    return prob

def wrapper_norm_rvs(x, n_sample_members):
    return stats.norm(loc=x[0], scale=x[1]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_norm_cdf(x):
    return stats.norm.cdf(x=x[2], loc=x[0], scale=x[1])

def wrapper_truncnorm_rvs(x, n_sample_members):
    return stats.truncnorm(a=-x[0]/x[1], b=_LARGE_NUMBER_, loc=x[0], scale=x[1]).rvs([n_sample_members] + list(x[0].shape))
def wrapper_truncnorm_cdf(x):
    return stats.truncnorm.cdf(x=x[2], loc=x[0], scale=x[1], a=-x[0]/x[1], b=_LARGE_NUMBER_)


#####################################
# Functions to sample distributions #
#####################################

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
        
        ens = _parallel_process(wrapper, iterable, total=mu.shape[parallelize_axis], name='truncnorm.rvs' if truncated else 'norm.rvs')
            
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


def sample_dist_gamma(unshifted_mu, sigma, shift, n_sample_members=15, move_axis=-1, truncated=False):
    
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
        if truncated:
            ens = truncgamma_gen()(a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, scale=scale, loc=shift).rvs([n_sample_members] + arr_shape)
        else:
            ens = stats.gamma(a=shape, scale=scale, loc=shift).rvs([n_sample_members] + arr_shape)
        
    else:
        parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
    
        iterable = zip(
            np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
            np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
            np.split(shift, shift.shape[parallelize_axis], parallelize_axis),
        )
        
        if truncated:
            wrapper = partial(wrapper_truncgamma_rvs, n_sample_members=n_sample_members)
        else:
            wrapper = partial(wrapper_gamma_rvs, n_sample_members=n_sample_members)
        
        ens = _parallel_process(wrapper, iterable, total=shape.shape[parallelize_axis], name='truncgamma.rvs' if truncated else 'gamma.rvs')
        
    # Move the ensemble axis somewhere else if the first position is not desired
    if move_axis != 0:
        ens = np.moveaxis(ens, 0, move_axis)
    
    return ens


##############################
# Functions to calculate CDF #
##############################

def cdf_gaussian(mu, sigma, over=None, below=None, truncated=False):
        
    assert isinstance(truncated, bool)
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if below is None:
        return 1 - cdf_gaussian(mu, sigma, over=None, below=over, truncated=truncated)
    
    else:
        cores = int(os.environ['pyanen_tqdm_workers'])        
        
        if isinstance(below, np.ndarray):
            assert below.shape == mu.shape
        elif isinstance(below, np.float) or isinstance(below, np.int):
            below = np.full(mu.shape, below)
        else:
            raise RuntimeError('below should be either a scalar or an Numpy array. Got {}'.format(type(below)))
        
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
                np.split(below, below.shape[parallelize_axis], parallelize_axis),
            )
            
            if truncated:
                probs = _parallel_process(wrapper_truncnorm_cdf, iterable, total=mu.shape[parallelize_axis], name='truncnorm.cdf' if truncated else 'norm.cdf')
            else:
                probs = _parallel_process(wrapper_norm_cdf, iterable, total=mu.shape[parallelize_axis], name='truncnorm.cdf' if truncated else 'norm.cdf')
            
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    
    return probs


def cdf_gamma(unshifted_mu, sigma, shift, over=None, below=None, truncated=False):
    
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if below is None:
        return 1 - cdf_gamma(unshifted_mu, sigma, shift, over=None, below=over, truncated=truncated)
    
    else:
        cores = int(os.environ['pyanen_tqdm_workers'])
        
        # Calculate distribution parameters
        shape = (unshifted_mu / sigma) ** 2
        scale = sigma ** 2 / unshifted_mu
            
        if isinstance(below, np.ndarray):
            assert below.shape == shape.shape
        elif isinstance(below, np.float) or isinstance(below, np.int):
            below = np.full(shape.shape, below)
        else:
            raise RuntimeError('below should be either a scalar or an Numpy array. Got {}'.format(type(below)))
        
        if cores == 1:
            if truncated:
                probs = truncgamma_gen().cdf(x=below, a=(-shift)/scale, b=_LARGE_NUMBER_, s=shape, loc=shift, scale=scale)
            else:
                probs = stats.gamma.cdf(x=below, a=shape, loc=shift, scale=scale)
                
        else:
            
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
            
            iterable = zip(
                np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
                np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
                np.split(shift, shift.shape[parallelize_axis], parallelize_axis),
                np.split(below, below.shape[parallelize_axis], parallelize_axis),
            )
            
            if truncated:
                probs = _parallel_process(wrapper_truncgamma_cdf, iterable, total=shape.shape[parallelize_axis], name='truncgamma.cdf' if truncated else 'gamma.cdf')
            else:
                probs = _parallel_process(wrapper_gamma_cdf, iterable, total=shape.shape[parallelize_axis], name='truncgamma.cdf' if truncated else 'gamma.cdf')
                
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    
    return probs


def cdf_gamma_hurdle(pop, mu, sigma, over=None, below=None):
    # The calculation is as follows:
    #
    # 1 - pop                                  given below=0
    # [1-pop] + pop*CDFgamma(y,shape,scale).   given below>0
    #
    
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if below is None:
        return 1 - cdf_gamma_hurdle(pop, mu, sigma, over=None, below=over)
    
    else:    
        # Calculate distribution parameters
        shape = (mu / sigma) ** 2
        scale = sigma ** 2 / mu
        
        cores = int(os.environ['pyanen_tqdm_workers'])
            
        if isinstance(below, np.ndarray):
            assert below.shape == shape.shape
        elif isinstance(below, np.float) or isinstance(below, np.int):
            below = np.full(shape.shape, below)
        else:
            raise RuntimeError('below should be either a scalar or an Numpy array. Got {}'.format(type(below)))
        
        if cores == 1:
            probs = wrapper_gamma_hurdle_cdf([pop, shape, scale, below])
                
        else:
            
            parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
            
            iterable = zip(
                np.split(pop, pop.shape[parallelize_axis], parallelize_axis),
                np.split(shape, shape.shape[parallelize_axis], parallelize_axis),
                np.split(scale, scale.shape[parallelize_axis], parallelize_axis),
                np.split(below, below.shape[parallelize_axis], parallelize_axis),
            )
            
            probs = _parallel_process(wrapper_gamma_hurdle_cdf, iterable, total=shape.shape[parallelize_axis], name='gammahurdle.cdf')
                
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    
    return probs
