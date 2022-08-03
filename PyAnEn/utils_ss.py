
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
# Date of Creation: 2022/08/01                              #
#############################################################
#
# Utility functions for calculating binned spread-skill correlation
#
# The steps to calculate the binned spread-skill correlation are summarized as follow:
#
#
# 1. "Average" ensemble spread calculation:
#    - for every ensemble prediction (e.g., across different days and stations)
#      and at a given valid time (e.g., 00 UTC) you compute the variance of the ensemble;
#    - you bin the variance values in intervals of equal size, or in intervals
#      with the same number of points;
#    - you do the average of the variance in each bin;
#    - you do the square root of the average variance in each bin.
# 
# 2. RMSE of the ensemble mean calculation:
#   - you compute the mean square error (MSE) of the ensemble mean across all
#     the available ensemble mean/obs pairs in each bin;
#   - you do the square root of [MSE multiplied by n/(n+1)], where n is
#     the number of members;
#
# 3. You plot RMSE v.s. spread
#


import os
import warnings

import numpy as np

from distutils import util
from tqdm.auto import tqdm


def _binned_spread_skill_create_split(variance, sq_error, nbins, sample_axis):
    
    if sample_axis is None:
        sample_axis = list(range(len(variance.shape)))
    
    assert variance.shape == sq_error.shape
    assert hasattr(sample_axis, '__len__'), 'sample_axis must be an array-like object!'

    # Move sample axis to the beginning
    to_axis = 0

    for from_axis in np.sort(sample_axis):
        if from_axis != to_axis:
            variance = np.moveaxis(variance, from_axis, to_axis)
            sq_error = np.moveaxis(sq_error, from_axis, to_axis)
        to_axis += 1

    shape_to_keep = variance.shape[to_axis:]
    
    # Flatten the sample dimensions and the kept dimensions separately
    variance = variance.reshape(-1, np.prod(shape_to_keep).astype(int))
    sq_error = sq_error.reshape(-1, np.prod(shape_to_keep).astype(int))
    
    # Sort arrays based on variance 
    for i in range(variance.shape[1]):
        index_sort = np.argsort(variance[:, i])
        variance[:, i] = variance[index_sort, i]
        sq_error[:, i] = sq_error[index_sort, i]
    
    # Combine and split arrays
    # The first column is absolute error
    # The second column is standard deviation
    #
    assert nbins <= sq_error.shape[0], 'Too many bins ({}) and too few samples ({})'.format(nbins, sq_error.shape[0])
    
    if nbins * 10 > sq_error.shape[0]:
        warnings.warn('{} samples and {} bins. Some bins have fewer than 10 samples.'.format(nbins, sq_error.shape[0]))
        
    arr_split = np.array_split(np.stack([variance, sq_error], axis=1), nbins, 0)
    
    return arr_split, shape_to_keep


def _binned_spread_skill_agg_no_boot(arr_split, reconstruct_shape, skip_nan=None):
    
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    variances = []
    mses = []
    
    for arr in tqdm(arr_split,
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='Spread skill aggregation'):
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        
        if skip_nan:
            variances.append([np.nanmean(arr[:, 0, i]) for i in range(arr.shape[2])])
            mses.append([np.nanmean(arr[:, 1, i]) for i in range(arr.shape[2])])
        else:
            variances.append([arr[:, 0, i].mean() for i in range(arr.shape[2])])
            mses.append([arr[:, 1, i].mean() for i in range(arr.shape[2])])
    
    variances = np.array(variances)
    mses = np.array(mses)

    assert variances.shape[1] == mses.shape[1] == np.prod(reconstruct_shape)

    spread = np.sqrt(variances)
    rmse = np.sqrt(mses)
    
    spread = spread.reshape(spread.shape[0], *reconstruct_shape)
    rmse = rmse.reshape(rmse.shape[0], *reconstruct_shape)

    # Transpose dimensions so that the dimensions to keep are at the beginning
    spread = np.moveaxis(spread, 0, -1)
    rmse = np.moveaxis(rmse, 0, -1)
    
    return spread, rmse


def _binned_spread_skill_agg_boot(arr_split, reconstruct_shape,
                                  n_samples=None, repeats=None, confidence=None, skip_nan=None):
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    variances_ci = []
    mses_ci = []
    
    nbins = len(arr_split)
    
    for arr in tqdm(arr_split,
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='Spread skill bootstraping'):
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        pop_size = arr.shape[0]
        
        for dim_i in range(arr.shape[2]):
            variances = []
            mses = []
            
            for _ in range(repeats):
                idx = np.random.randint(pop_size, size=n_samples)
                
                if skip_nan:
                    variances.append(np.nanmean(arr[idx, 0, dim_i]))
                    mses.append(np.nanmean(arr[idx, 1, dim_i]))
                else:
                    variances.append(arr[idx, 0, dim_i].mean())
                    mses.append(arr[idx, 1, dim_i].mean())
            
            variance_ci = np.quantile(variances, [1-confidence, confidence])
            mse_ci = np.quantile(mses, [1-confidence, confidence])
            
            variances_ci.append([variance_ci[0], np.mean(variances), variance_ci[1]])
            mses_ci.append([mse_ci[0], np.mean(mses), mse_ci[1]])
        
    variances_ci = np.array(variances_ci)
    mses_ci = np.array(mses_ci)
    
    assert len(variances_ci.shape) == len(mses_ci.shape) == 2
    assert variances_ci.shape[0] == mses_ci.shape[0] == np.prod(reconstruct_shape) * nbins

    spread_ci = np.sqrt(variances_ci)
    rmse_ci = np.sqrt(mses_ci)
    
    spread_ci = spread_ci.reshape(nbins, *reconstruct_shape, 3)
    rmse_ci = rmse_ci.reshape(nbins, *reconstruct_shape, 3)
    
    # Transpose dimensions so that the dimensions to keep are at the beginning
    spread_ci = np.moveaxis(spread_ci, 0, -2)
    rmse_ci = np.moveaxis(rmse_ci, 0, -2)
    
    return spread_ci, rmse_ci


def binned_spread_skill(
    variance, sq_error, nbins, sample_axis,
    boot_samples=None, repeats=None, confidence=None, skip_nan=None):
    
    arr_split, shape_to_keep = _binned_spread_skill_create_split(
        variance=variance, sq_error=sq_error,
        nbins=nbins, sample_axis=sample_axis)
    
    if boot_samples is None:
        return _binned_spread_skill_agg_no_boot(
            arr_split=arr_split, reconstruct_shape=shape_to_keep,
            skip_nan=skip_nan)
    else:
        return _binned_spread_skill_agg_boot(
            arr_split=arr_split, reconstruct_shape=shape_to_keep,
            n_samples=boot_samples, repeats=repeats, confidence=confidence,
            skip_nan=skip_nan)
