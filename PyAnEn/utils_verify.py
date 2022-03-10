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
# Utility functions for verifications
#

import os
import warnings

import numpy as np
import scipy.stats as st

from distutils import util
from tqdm.auto import tqdm
from sklearn import metrics
from scipy import stats, special
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity


def rankdata_wrapper(x, pbar=None):
    if pbar is not None:
        pbar.update(1)
    return rankdata(x, method='min')


def rank_histogram(f, o, ensemble_axis):
    # Reference:
    # https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        o_mask = np.where(np.isnan(o))
    else:
        assert not np.any(np.isnan(f)), "[rank_histogram] f has NANs. Try to set os.environ['pyanen_skip_nan']='True'"
        assert not np.any(np.isnan(o)), "[rank_histogram] o has NANs. Try to set os.environ['pyanen_skip_nan']='True'"
    
    # Move the ensemble axis to the first axis and
    # combine observations and ensembles along the first axis
    #
    c = np.vstack((
        np.expand_dims(o, 0),
        np.moveaxis(f, ensemble_axis, 0)))
    
    # Calculate ranks for ensemble members
    with tqdm(total=np.prod(np.delete(c.shape, 0)),
              disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
              leave=util.strtobool(os.environ['pyanen_tqdm_leave'])) as pbar:
        ranks = np.apply_along_axis(rankdata_wrapper, 0, c, pbar=pbar)
    
    # Retrieves observation ranks
    obs_ranks = ranks[0]
    
    # Check for ties
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    tie = np.unique(ties)
    
    # For ties, randomly decide which bin the observation should go to
    for i in range(1, len(tie)):
        index = obs_ranks[ties == tie[i]]
        obs_ranks[ties == tie[i]] = [np.random.randint(index[j], index[j] + tie[i] + 1, tie[i])[0] 
                                     for j in range(len(index))]
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        obs_ranks[o_mask] = np.nan
        
    return obs_ranks


def ens_to_prob_kde(ens, over=None, below=None, bandwidth=None, kernel=None,
                    samples=None, spread_multiplier=None, pbar=None):
    
    if bandwidth is None: bandwidth = float(os.environ['pyanen_kde_bandwidth'])
    if kernel is None: kernel = os.environ['pyanen_kde_kernel']
    if samples is None: samples = int(os.environ['pyanen_kde_samples'])
    if spread_multiplier is None: spread_multiplier = int(os.environ['pyanen_kde_multiply_spread'])
    
    ens = ens.reshape(-1, 1)
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(ens)
    
    vmin, vmax = ens.min(), ens.max()
    spread = vmax - vmin
    
    x = np.linspace(vmin - spread * spread_multiplier, below, samples) \
        if over is None else np.linspace(over, vmax + spread * spread_multiplier, samples)
        
    interval = x[1] - x[0]
    x = x.reshape(-1, 1)
    
    if pbar is not None:
        pbar.update(1)
        
    return np.sum(np.exp(kde.score_samples(x))) * interval


def ens_to_prob(f, ensemble_aixs, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    with tqdm(total=np.prod(np.delete(f.shape, ensemble_aixs)),
              disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
              leave=util.strtobool(os.environ['pyanen_tqdm_leave'])) as pbar:
        ret = np.apply_along_axis(ens_to_prob_kde, ensemble_aixs, f, over=over, below=below, pbar=pbar)
    return ret
    

def binarize_obs(o, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    return o < below if over is None else o > over
        
        
def calculate_reliability(f_prob, o_binary, nbins):
    assert f_prob.shape == o_binary.shape
    
    # Flatten
    f_prob = f_prob.flatten()
    o_binary = o_binary.flatten()
    
    # Sort based on forecasted probability
    index_sort = np.argsort(f_prob)
    f_prob = f_prob[index_sort]
    o_binary = o_binary[index_sort]
    
    # Combine and split arrays
    assert nbins <= f_prob.shape[0], 'Too many bins ({}) and too few samples ({})'.format(nbins, f_prob.shape[0])
    
    if nbins * 10 > f_prob.shape[0]:
        warnings.warn('{} samples and {} bins. Some bins have fewer than 10 samples.'.format(f_prob.shape[0], nbins))
        
    arr_split = np.array_split(np.stack([f_prob, o_binary], axis=1), nbins, 0)
    
    # Calculate average forecasted probability and the average observed frequency
    if util.strtobool(os.environ['pyanen_skip_nan']):
        rel = np.array([np.nanmean(i, axis=0) for i in arr_split])
    else:
        rel = np.array([np.mean(i, axis=0) for i in arr_split])
        
    forecasted, observed = rel[:, 0], rel[:, 1]
    
    return forecasted, observed


def calculate_roc(f_prob, o_binary):
    # Flatten
    f_prob = f_prob.flatten()
    o_binary = o_binary.flatten()
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        mask = np.isnan(f_prob) or np.isnan(o_binary)
        f_prob = f_prob[~mask]
        o_binary = o_binary[~mask]
    
    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=o_binary, y_score=f_prob)
    auc = metrics.auc(fpr, tpr)
    
    return fpr, tpr, auc


##############################
# Functions for Bootstraping #
##############################

def boot_vec(pop, n_samples=None, repeats=None, confidence=None, pbar=None, skip_nan=False, return_delta=True):
    # Reference: https://www.cawcr.gov.au/projects/verification/BootstrapCIs.html
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    
    if skip_nan:
        boot_samples_mean = [np.nanmean(np.random.choice(pop, size=n_samples, replace=True)) for _ in range(repeats)]
    else:
        boot_samples_mean = [np.random.choice(pop, size=n_samples, replace=True).mean() for _ in range(repeats)]
        
    boot_samples_mean = np.array(boot_samples_mean)
    
    sample_mean = boot_samples_mean.mean()
    sample_std = boot_samples_mean.std()

    ci = st.t.interval(alpha=confidence, df=repeats-1, loc=sample_mean, scale=sample_std)
    
    if pbar is not None:
        pbar.update(1)
    
    if return_delta:
        return np.array([sample_mean, sample_mean - ci[0]])
    else:
        return np.array([ci[0], sample_mean, ci[1]])


def boot_arr(metric, sample_axis, n_samples=None, repeats=None, confidence=None):
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    
    if sample_axis is None:
        sample_axis = list(range(len(metric.shape)))
    
    # Move average axis to the beginning
    to_axis = 0

    for from_axis in np.sort(sample_axis):
        if from_axis != to_axis:
            metric = np.moveaxis(metric, from_axis, to_axis)
        to_axis += 1

    shape_to_keep = metric.shape[to_axis:]

    # Flatten the sample dimensions and the kept dimensions separately
    metric = metric.reshape(-1, np.prod(shape_to_keep).astype(int))
    assert len(metric.shape) == 2
    
    with tqdm(total=metric.shape[1],
              disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
              leave=util.strtobool(os.environ['pyanen_tqdm_leave'])) as pbar:
        intervals = np.apply_along_axis(boot_vec, 0, metric, n_samples=n_samples,
                                        repeats=repeats, confidence=confidence, pbar=pbar,
                                        skip_nan=util.strtobool(os.environ['pyanen_skip_nan']),
                                        return_delta=util.strtobool(os.environ['pyanen_boot_return_delta']))
        
    intervals = intervals.reshape(-1, *shape_to_keep)
    
    return intervals


##########################################
# Functions for Spread Skill Correlation #
##########################################

def _binned_spread_skill_create_split(variance, ab_error, nbins, sample_axis):
    
    assert variance.shape == ab_error.shape
    
    if sample_axis is None:
        sample_axis = list(range(len(variance.shape)))
    
    # Move average axis to the beginning
    to_axis = 0

    for from_axis in np.sort(sample_axis):
        if from_axis != to_axis:
            variance = np.moveaxis(variance, from_axis, to_axis)
            ab_error = np.moveaxis(ab_error, from_axis, to_axis)
        to_axis += 1

    shape_to_keep = variance.shape[to_axis:]
    
    # Flatten the sample dimensions and the kept dimensions separately
    variance = variance.reshape(-1, *shape_to_keep)
    ab_error = ab_error.reshape(-1, *shape_to_keep)
    
    variance = variance.reshape(-1, np.prod(shape_to_keep).astype(int))
    ab_error = ab_error.reshape(-1, np.prod(shape_to_keep).astype(int))
    
    # Sort arrays based on error
    for i in range(variance.shape[1]):
        index_sort = np.argsort(variance[:, i])
        variance[:, i] = variance[index_sort, i]
        ab_error[:, i] = ab_error[index_sort, i]
    
    # Combine and split arrays
    # The first column is absolute error
    # The second column is standard deviation
    #
    assert nbins <= ab_error.shape[0], 'Too many bins ({}) and too few samples ({})'.format(nbins, ab_error.shape[0])
    
    if nbins * 10 > ab_error.shape[0]:
        warnings.warn('{} samples and {} bins. Some bins have fewer than 10 samples.'.format(nbins, ab_error.shape[0]))
        
    arr_split = np.array_split(np.stack([np.sqrt(variance), ab_error], axis=1), nbins, 0)
    
    return arr_split, shape_to_keep


def _binned_spread_skill_agg_no_boot(arr_split, reconstruct_shape, skip_nan=None):
    
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    spreads = []
    errors = []
    
    for arr in tqdm(arr_split,
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave'])):
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        
        if skip_nan:
            spreads.append([np.nanmean(arr[:, 0, i]) for i in range(arr.shape[2])])
            errors.append([np.nanmean(arr[:, 1, i]) for i in range(arr.shape[2])])
        else:
            spreads.append([arr[:, 0, i].mean() for i in range(arr.shape[2])])
            errors.append([arr[:, 1, i].mean() for i in range(arr.shape[2])])
    
    spreads = np.array(spreads)
    errors = np.array(errors)
    
    assert errors.shape[1] == spreads.shape[1] == np.prod(reconstruct_shape)
    
    spreads = spreads.reshape(spreads.shape[0], *reconstruct_shape)
    errors = errors.reshape(errors.shape[0], *reconstruct_shape)
    
    return spreads, errors


def _binned_spread_skill_agg_boot(arr_split, reconstruct_shape,
                                  n_samples=None, repeats=None, confidence=None, skip_nan=None):
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    spreads_ci = []
    errors_ci = []
    
    nbins = len(arr_split)
    
    for arr in tqdm(arr_split,
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave'])):
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        pop_size = arr.shape[0]
        
        for dim_i in range(arr.shape[2]):
            spreads = []
            errors = []
            
            for _ in range(repeats):
                idx = np.random.randint(pop_size, size=n_samples)
                
                if skip_nan:
                    spreads.append(np.nanmean(arr[idx, 0, dim_i]))
                    errors.append(np.nanmean(arr[idx, 1, dim_i]))
                else:
                    spreads.append(arr[idx, 0, dim_i].mean())
                    errors.append(arr[idx, 1, dim_i].mean())
            
            spread_mean, spread_std = np.mean(spreads), np.std(spreads)
            error_mean, error_std = np.mean(errors), np.std(errors)
            
            spread_ci = st.t.interval(alpha=confidence, df=repeats-1, loc=spread_mean, scale=spread_std)
            error_ci = st.t.interval(alpha=confidence, df=repeats-1, loc=error_mean, scale=error_std)
            
            spreads_ci.append([spread_ci[0], spread_mean, spread_ci[1]])
            errors_ci.append([error_ci[0], error_mean, error_ci[1]])
        
    spreads_ci = np.array(spreads_ci)
    errors_ci = np.array(errors_ci)
    
    assert len(spreads_ci.shape) == len(errors_ci.shape) == 2, 'spreads_ci: {}, errors_ci: {}'.format(spreads_ci.shape, errors_ci.shape)
    assert spreads_ci.shape[0] == errors_ci.shape[0] == np.prod(reconstruct_shape) * nbins, \
        'Got {}, {}, {}'.format(spreads_ci.shape[0], errors.shape[0], np.prod(reconstruct_shape) * nbins)
    
    spreads_ci = spreads_ci.reshape(nbins, *reconstruct_shape, 3)
    errors_ci = errors_ci.reshape(nbins, *reconstruct_shape, 3)
    
    spreads_ci = np.moveaxis(spreads_ci, -1, 0)
    errors_ci = np.moveaxis(errors_ci, -1, 0)
    
    return spreads_ci, errors_ci


def binned_spread_skill(
    variance, ab_error, nbins, sample_axis,
    boot_samples=None, repeats=None, confidence=None, skip_nan=None):
    
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    
    arr_split, shape_to_keep = _binned_spread_skill_create_split(
        variance=variance, ab_error=ab_error,
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
        
        
######################
# Functions for CRPS #
######################


def _lbeta(x1, x2):
    log_prod_gamma_x = np.log(special.gamma(x1)) + np.log(special.gamma(x2))
    log_gamma_sum_x = np.log(special.gamma(x1 + x2))
    return log_prod_gamma_x - log_gamma_sum_x


def crps_csgd(mu, sigma, shift, obs, reduce_sum=True):
    
    # Calculate distribution parameters
    shape = np.square(mu / sigma)
    scale = (np.square(sigma)) / mu
    
    # First term in Eq. (5)
    y_bar = (obs - shift) / scale
    
    # F_k_y = tf.math.igamma(shape, 1. * y_bar)
    F_k_y = stats.gamma.cdf(1. * y_bar, shape)
    
    c_bar = (-1 * shift) / scale
    
    # Second term in Eq. (5)
    if util.strtobool(os.environ['pyanen_use_tensorflow_math']):
        import tensorflow as tf
        lbeta_ret = tf.math.lbeta(tf.stack([np.full(mu.shape, 0.5), shape + 0.5], axis=len(shape.shape))).numpy()
        F_2k_2c = tf.math.igamma(2. * shape, 1. * 2. * c_bar).numpy()
        F_k_c = tf.math.igamma(shape, 1. * c_bar).numpy()
        F_kp1_y = tf.math.igamma(shape+1., 1. * y_bar).numpy()
        F_kp1_c = tf.math.igamma(shape+1., 1. * c_bar).numpy()
    else:
        lbeta_ret = _lbeta(np.full(mu.shape, 0.5), shape + 0.5)
        F_2k_2c = stats.gamma.cdf(1. * 2. * c_bar, 2. * shape)
        F_k_c = stats.gamma.cdf(1. * c_bar, shape)
        F_kp1_y = stats.gamma.cdf(1. * y_bar, shape+1.)
        F_kp1_c = stats.gamma.cdf(1. * c_bar, shape+1.)
        
    B_05_kp05 = np.exp(lbeta_ret)
    
    c1 = y_bar * (2. * F_k_y - 1.)
    c2 = shape * (2. * F_kp1_y - 1. + np.square(F_k_c) - 2. * F_kp1_c * F_k_c)
    c3 = c_bar * np.square(F_k_c)
    c4 = (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    
    crps = c1 - c2 - c3 - c4
    
    CRPS = crps * scale
    
    if reduce_sum:
        if util.strtobool(os.environ['pyanen_skip_nan']):
            return np.nanmean(CRPS)
        else:
            return np.mean(CRPS)
    else:
        return CRPS
