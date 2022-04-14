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
import pandas as pd

from scipy import stats
from distutils import util
from tqdm.auto import tqdm
from sklearn import metrics
from functools import partial
from sklearn.neighbors import KernelDensity
from tqdm.contrib.concurrent import process_map


def rankdata(x):
    ranks = stats.rankdata(x, method='min', axis=0)
    
    obs_ranks = ranks[0]
    
    # Check for ties
    ties = np.nansum(ranks[0] == ranks[1:], axis=0)
    unique_tie = np.unique(ties)
    unique_tie = unique_tie[~np.isnan(unique_tie)]
    
    # For ties, randomly decide which bin the observation should go to
    for i in range(1, len(unique_tie)):
        
        index = obs_ranks[ties == unique_tie[i]]
        obs_ranks[ties == unique_tie[i]] = [np.random.randint(index[j], index[j] + unique_tie[i] + 1, unique_tie[i])[0] 
                                            for j in range(len(index))]
    
    return obs_ranks


def rank_histogram(f, o, ensemble_axis):
    # Reference:
    # https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        final_mask = np.where(np.isnan(o) | np.isnan(f.sum(axis=ensemble_axis)))
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
    cores = int(os.environ['pyanen_tqdm_workers'])
    
    if cores == 1:
        obs_ranks = rankdata(c)
    else:
        parallelize_axis = int(os.environ['pyanen_tqdm_map_axis'])
        chunksize = int(os.environ['pyanen_tqdm_chunksize'])
        assert parallelize_axis < 0, 'parallelize_axis needs to be negative, counting from the end of dimensions, excluding the ensemble axis'
        
        ranks = process_map(rankdata,
                            np.split(c, c.shape[parallelize_axis], parallelize_axis),
                            disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                            leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                            chunksize=chunksize, max_workers=cores,
                            desc='Rank histogram')
        
        obs_ranks = np.concatenate(ranks, axis=parallelize_axis)
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        obs_ranks = obs_ranks.astype(np.float)
        obs_ranks[final_mask] = np.nan
        
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


def ens_to_prob_moments(ens, over=None, below=None, pbar=None, axis=None):
    
    if pbar is not None:
        pbar.update(1)
    
    if over is None:
        return np.count_nonzero(ens <= below, axis=axis) / (len(ens) if axis is None else ens.shape[axis])
    else:
        return np.count_nonzero(ens >= over, axis=axis) / (len(ens)  if axis is None else ens.shape[axis])


def ens_to_prob(f, ensemble_aixs, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    cores = int(os.environ['pyanen_tqdm_workers'])
    
    if cores == 1:
        if os.environ['pyanen_ens_to_prob_method'] == 'kde':
            with tqdm(total=np.prod(np.delete(f.shape, ensemble_aixs)),
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='KDE') as pbar:
                
                ret = np.apply_along_axis(ens_to_prob_kde, ensemble_aixs, f, over=over, below=below, pbar=pbar)
                
        elif os.environ['pyanen_ens_to_prob_method'] == 'moments':
            ret = ens_to_prob_moments(f, over=over, below=below, axis=ensemble_aixs)
            
        else:
            msg = 'Unknown method for converting ensembles to probability. ' + \
                'Got {}. Expect one of [kde, moments]'.format(os.environ['pyanen_ens_to_prob_method'])
            raise Exception(msg)
            
    else:
        if os.environ['pyanen_ens_to_prob_method'] == 'kde':
            transfer_func = ens_to_prob_kde
        elif os.environ['pyanen_ens_to_prob_method'] == 'moments':
            transfer_func = ens_to_prob_moments
        else:
            msg = 'Unknown method for converting ensembles to probability. ' + \
                'Got {}. Expect one of [kde, moments]'.format(os.environ['pyanen_ens_to_prob_method'])
            raise Exception(msg)
        
        # Move axis
        f = np.moveaxis(f, ensemble_aixs, 0)
        
        n_members = f.shape[0]
        f_shape = f.shape[1:]
        
        f = f.reshape(n_members, -1)
        
        chunksize = 1000 # int(os.environ['pyanen_tqdm_chunksize'])
        parallelize_axis = -1 # int(os.environ['pyanen_tqdm_map_axis'])
        assert parallelize_axis == -1, 'parallelize_axis needs to be -1 for the ens_to_prob operation. Got {}'.format(parallelize_axis)
        
        ret = process_map(partial(transfer_func, over=over, below=below),
                          np.split(f, f.shape[parallelize_axis], parallelize_axis),
                          disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                          leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                          chunksize=chunksize, max_workers=cores,
                          desc='Ensemble to probability')
        
        ret = np.array(ret).reshape(f_shape)
            
    return ret


def binarize_obs(o, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    return o < below if over is None else o > over


def calculate_roc(f_prob, o_binary):
    # Flatten
    f_prob = f_prob.flatten()
    o_binary = o_binary.flatten()
    
    if util.strtobool(os.environ['pyanen_skip_nan']):
        mask = np.isnan(f_prob) | np.isnan(o_binary)
        f_prob = f_prob[~mask]
        o_binary = o_binary[~mask]
    
    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=o_binary, y_score=f_prob)
    auc = metrics.auc(fpr, tpr)
    
    return fpr, tpr, auc


##############################
# Functions for Bootstraping #
##############################

def boot_vec(pop, n_samples=None, repeats=None, confidence=None, pbar=None, skip_nan=False):
    # Reference: https://www.cawcr.gov.au/projects/verification/BootstrapCIs.html
    assert len(pop.shape) == 1, 'pop must be a vector'
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    
    if skip_nan:
        boot_samples_mean = [np.nanmean(np.random.choice(pop, size=n_samples, replace=True)) for _ in range(repeats)]
    else:
        boot_samples_mean = [np.random.choice(pop, size=n_samples, replace=True).mean() for _ in range(repeats)]
        
    boot_samples_mean = np.array(boot_samples_mean)
    
    sample_mean = boot_samples_mean.mean()
    ci = np.quantile(boot_samples_mean, [1-confidence, confidence])
    
    # sample_std = boot_samples_mean.std()
    # ci = st.t.interval(alpha=confidence, df=repeats-1, loc=sample_mean, scale=sample_std)
    
    if pbar is not None:
        pbar.update(1)
    
    # Return min, mean, max of error bars
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
              leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
              desc='Bootstraping') as pbar:
        intervals = np.apply_along_axis(boot_vec, 0, metric, n_samples=n_samples,
                                        repeats=repeats, confidence=confidence, pbar=pbar,
                                        skip_nan=util.strtobool(os.environ['pyanen_skip_nan']))
        
    intervals = intervals.reshape(-1, *shape_to_keep)
    
    return intervals


#####################################
# Functions for reliability diagram #
#####################################
#
# Code referenced from sklearn
# https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/calibration.py#L869
#

def _reliability_split(f_prob, o_binary, nbins):
    
    assert f_prob.shape == o_binary.shape
    
    # Flatten
    f_prob = f_prob.flatten()
    o_binary = o_binary.flatten()
    
    # Split array for reliability diagram
    bins = np.linspace(0.0, 1.0 + 1e-8, nbins + 1)
    binids = np.digitize(f_prob, bins) - 1
    
    return binids, f_prob, o_binary, bins

def _reliability_agg_no_boot(binids, y_prob, y_true, bins):

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    counts = pd.value_counts(binids)
    counts.name = 'Samples in each bin'
    
    return prob_pred, prob_true, counts.sort_index()

def _reliability_agg_boot(binids, y_prob, y_true, bins,
                          n_samples=None, repeats=None, confidence=None, skip_nan=None):
    
    if n_samples is None: n_samples = int(os.environ['pyanen_boot_samples'])
    if repeats is None: repeats = int(os.environ['pyanen_boot_repeats'])
    if confidence is None: confidence = float(os.environ['pyanen_boot_confidence'])
    if skip_nan is None: skip_nan = util.strtobool(os.environ['pyanen_skip_nan'])
    
    probs_pred = []
    probs_true = []
    
    for binid in range(len(bins)):
        indices = np.where(binids == binid)[0]

        if len(indices) > 0:
            sample_probs = []
            sample_trues = []

            for _ in range(repeats):
                random_indices = np.random.choice(indices, size=n_samples, replace=True)

                if skip_nan:
                    sample_probs.append(np.nanmean(y_prob[random_indices]))
                    sample_trues.append(np.nanmean(y_true[random_indices]))
                else:
                    sample_probs.append(np.mean(y_prob[random_indices]))
                    sample_trues.append(np.mean(y_true[random_indices]))

            ci = np.quantile(sample_probs, [1-confidence, confidence])
            probs_pred.append([ci[0], np.mean(sample_probs), ci[1]])

            ci = np.quantile(sample_trues, [1-confidence, confidence])
            probs_true.append([ci[0], np.mean(sample_trues), ci[1]])

        else:
            probs_pred.append([])
            probs_true.append([])

    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0
    
    probs_pred = [probs_pred[i] for i in range(len(bins)) if nonzero[i]]
    probs_true = [probs_true[i] for i in range(len(bins)) if nonzero[i]]

    counts = pd.value_counts(binids)
    counts.name = 'Samples in each bin'

    return np.array(probs_pred), np.array(probs_true), counts.sort_index()


def reliability_diagram(f_prob, o_binary, nbins, 
                        boot_samples=None, repeats=None,
                        confidence=None, skip_nan=None):
    
    binids, f_prob, o_binary, bins = _reliability_split(
        f_prob=f_prob, o_binary=o_binary, nbins=nbins)
    
    if boot_samples is None:
        return _reliability_agg_no_boot(
            binids, f_prob, o_binary, bins)
    else:
        return _reliability_agg_boot(
            binids, f_prob, o_binary, bins,
            boot_samples, repeats, confidence, skip_nan)
    

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
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='Spread skill aggregation'):
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
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='Spread skill bootstraping'):
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
            
            spread_mean, error_mean  = np.mean(spreads), np.mean(errors)
            spread_ci = np.quantile(spreads, [1-confidence, confidence])
            error_ci = np.quantile(errors, [1-confidence, confidence])
            
            # spread_mean, spread_std = np.mean(spreads), np.std(spreads)
            # error_mean, error_std = np.mean(errors), np.std(errors)
            # spread_ci = st.t.interval(alpha=confidence, df=repeats-1, loc=spread_mean, scale=spread_std)
            # error_ci = st.t.interval(alpha=confidence, df=repeats-1, loc=error_mean, scale=error_std)
            
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


#####################
# Functions for IOU #
#####################

def _iou(f_binary, o_binary, axis=None):
    assert np.all(~np.isnan(f_binary)) and np.all(~np.isnan(o_binary))
    
    # Calculate components
    intersect = f_binary & o_binary
    union = f_binary | o_binary
    
    # IOU
    iou = np.count_nonzero(intersect, axis=axis) / np.count_nonzero(union, axis=axis)
    return iou


def iou_determ(f, o, axis=None, over=None, below=None):
    assert f.shape == o.shape
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if over is None:
        f_binary = f < below
        o_binary = o < below
    else:
        f_binary = f > over
        o_binary = o > over
    
    return _iou(f_binary, o_binary, axis)


def iou_prob(f_prob, o_binary, axis=None, over=None, below=None):
    assert f_prob.shape == o_binary.shape
    assert (over is None) ^ (below is None), 'Must specify over or below'
    
    if over is None:
        assert 0 <= below <= 1, 'below needs to be a probability in [0, 1]'
        f_binary = f_prob < below
    else:
        assert 0 <= over <= 1, 'over needs to be a probability in [0, 1]'
        f_binary = f_prob > over
    
    return _iou(f_binary, o_binary, axis)
