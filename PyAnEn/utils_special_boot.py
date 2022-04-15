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
# Date of Creation: 2022/04/14                              #
#############################################################
#
# Utility functions for correlation
#

import os

import numpy as np

from distutils import util
from tqdm.auto import tqdm


def corr(f, o, avg_axis=None, n_samples=None, repeats=None, confidence=None, disable=None, leave=None, skip_nan=None):
    return _workflow(f, o, 'Correlation', _func_corr, avg_axis, n_samples, repeats, confidence, disable, leave, skip_nan)


def brier_decomp(f, o, avg_axis=None, n_samples=None, repeats=None, confidence=None, disable=None, leave=None, skip_nan=None):
    ret = _workflow(f, o, 'Brier decomposition', _func_brier_decomp, avg_axis, n_samples, repeats, confidence, disable, leave, skip_nan)
    
    if n_samples is None:
        assert ret.shape[0] == 3
        
        rel, res, unc = ret
        return rel, res, unc
    
    else:
        assert ret.shape[0] == 9
        
        ret = ret.reshape(3, 3, *ret.shape[1:])
        rel, res, unc = ret
        return rel, res, unc


######################
# Internal Functions #
######################

def _func_brier_decomp(f_slice, o_slice, n_samples, repeats, confidence, skip_nan, n_bins=19):
    
    bins = np.linspace(0, 1, n_bins + 2)[1:-1]
    belongs = np.digitize(f_slice, bins)

    if skip_nan:
        o_bar = np.nanmean(o_slice)
    else:
        o_bar = np.mean(o_slice)
    
    if n_samples is None:

        reliability = 0
        resolution = 0

        for bin_i in range(belongs.max() + 1):
            mask = belongs == bin_i
            N_i = np.count_nonzero(mask)

            if N_i != 0:
                if skip_nan:
                    reliability += N_i * (np.nanmean(f_slice[mask]) - np.nanmean(o_slice[mask])) ** 2
                    resolution += N_i * (np.nanmean(o_slice[mask]) - o_bar) ** 2
                else:
                    reliability += N_i * (np.mean(f_slice[mask]) - np.mean(o_slice[mask])) ** 2
                    resolution += N_i * (np.mean(o_slice[mask]) - o_bar) ** 2

        reliability /= len(f_slice)
        resolution /= len(f_slice)
        uncertainty = o_bar * (1 - o_bar)
        
        return [reliability, resolution, uncertainty]
    
    else:
        
        reliability = []
        resolution = []
        uncertainty = []
        
        for _ in range(repeats):
            indices = np.random.choice(list(range(len(f_slice))), size=n_samples, replace=True)
            ret = _func_brier_decomp(f_slice[indices], o_slice[indices], n_samples=None,
                                     repeats=repeats, confidence=confidence, skip_nan=skip_nan)
            
            reliability.append(ret[0])
            resolution.append(ret[1])
            uncertainty.append(ret[2])
        
        if skip_nan:
            ci_rel = np.nanquantile(reliability, [1-confidence, confidence])
            ci_res = np.nanquantile(resolution, [1-confidence, confidence])
            ci_unc = np.nanquantile(uncertainty, [1-confidence, confidence])
            
            return [[ci_rel[0], np.nanmean(reliability), ci_rel[1]],
                    [ci_res[0], np.nanmean(resolution), ci_res[1]],
                    [ci_unc[0], np.nanmean(uncertainty), ci_unc[1]]]
        
        else:
            ci_rel = np.quantile(reliability, [1-confidence, confidence])
            ci_res = np.quantile(resolution, [1-confidence, confidence])
            ci_unc = np.quantile(uncertainty, [1-confidence, confidence])
            
            return [[ci_rel[0], np.mean(reliability), ci_rel[1]],
                    [ci_res[0], np.mean(resolution), ci_res[1]],
                    [ci_unc[0], np.mean(uncertainty), ci_unc[1]]]
            

def _func_corr(f_slice, o_slice, n_samples, repeats, confidence, skip_nan):
    
    if n_samples is None:
        if skip_nan:
            mask = (~np.isnan(f_slice)) & (~np.isnan(o_slice))
            return np.corrcoef(f_slice[mask], o_slice[mask])[0, 1]
        else:
            return np.corrcoef(f_slice, o_slice)[0, 1]
    
    else:
        corr = []
            
        for _ in range(repeats):
            indices = np.random.choice(list(range(len(f_slice))), size=n_samples, replace=True)
            corr.append(np.corrcoef(f_slice[indices], o_slice[indices])[0, 1])
            
        if skip_nan:
            ci = np.nanquantile(corr, [1-confidence, confidence])
            return [ci[0], np.nanmean(corr), ci[1]]
        else:
            ci = np.quantile(corr, [1-confidence, confidence])
            return [ci[0], np.mean(corr), ci[1]]
        

def _workflow(f, o, func_name, func, avg_axis=None, n_samples=None, repeats=None, confidence=None, disable=None, leave=None, skip_nan=None):
    
    # Assign default values if not set by function input
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    if disable is None: disable = util.strtobool(os.environ['pyanen_tqdm_disable'])
    if leave is None: leave = util.strtobool(os.environ['pyanen_tqdm_leave'])
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    assert f.shape == o.shape, 'Fatal error: shape mismatch {} != {}'.format(f.shape, o.shape)
    
    desc = func_name + ('' if n_samples is None else ' with boostrapping')
    
    if avg_axis is None: avg_axis = list(range(len(o.shape)))
    else: avg_axis = list(avg_axis)
    
    # Calculate the output shape indices
    out_shape = list(set(range(len(o.shape))) - set(avg_axis))

    o = o.transpose(out_shape + avg_axis)
    f = f.transpose(out_shape + avg_axis)
    
    # Separate the output dimensions from the flat dimensions    
    out_shape = f.shape[:len(out_shape)]
    flat_shape = f.shape[-len(avg_axis):]

    o = o.reshape(np.prod(out_shape).astype(int), np.prod(flat_shape).astype(int))
    f = f.reshape(np.prod(out_shape).astype(int), np.prod(flat_shape).astype(int))
    
    # At this point, the forecast and observation arrays have been reshaped to
    # two-dimensional arrays with [apply axis, aggregation axis]
    #
    
    ret = []
    
    for i in tqdm(range(f.shape[0]), disable=disable, leave=leave, desc=desc):
        f_slice = f[i].ravel()
        o_slice = o[i].ravel()
        ret.append(func(f_slice, o_slice, n_samples, repeats, confidence, skip_nan))
    
    # Reshape so that if bootstrapping is carried out, intervals are places along the first dimension
    ret = np.array(ret).reshape(*out_shape, -1)
    ret = np.moveaxis(ret, -1, 0)
    if ret.shape[0] == 1: ret = ret.squeeze(0)
    
    return ret
