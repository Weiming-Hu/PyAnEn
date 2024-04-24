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

import numpy as np
import pandas as pd

from scipy import stats
from distutils import util
from tqdm.auto import tqdm
from sklearn import metrics
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


def ens_to_prob_kde(ens, bandwidth=None, kernel=None,
                    samples=None, spread_multiplier=None, pbar=None):
    
    if bandwidth is None: bandwidth = float(os.environ['pyanen_kde_bandwidth'])
    if kernel is None: kernel = os.environ['pyanen_kde_kernel']
    if samples is None: samples = int(os.environ['pyanen_kde_samples'])
    if spread_multiplier is None: spread_multiplier = int(os.environ['pyanen_kde_multiply_spread'])
    
    ens = ens.ravel()
    below, ens = ens[0], ens[1:]
    ens = ens.reshape(-1, 1)
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(ens)
    
    vmin, vmax = ens.min(), ens.max()
    spread = vmax - vmin
    
    x = np.linspace(vmin - spread * spread_multiplier, below, samples) \
        
    interval = x[1] - x[0]
    x = x.reshape(-1, 1)
    
    if pbar is not None:
        pbar.update(1)
        
    return np.sum(np.exp(kde.score_samples(x))) * interval


def ens_to_prob_moments(ens, pbar=None):
    if pbar is not None: pbar.update(1)
    return np.count_nonzero(ens[1:] <= ens[0], axis=0) / (ens.shape[0] - 1)


def ens_to_prob(f, ensemble_aixs, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    if below is None: return 1 - ens_to_prob(f, ensemble_aixs, below=over)
    
    if isinstance(below, np.ndarray):
        assert below.shape == tuple(np.delete(f.shape, ensemble_aixs))
    else:
        assert not hasattr(below, '__len__'), 'below/over could either be an numpy array or a scalar'
        below = np.full(np.delete(f.shape, ensemble_aixs), below)
    
    # At this point, below has the same dimensions as f except for having length of 1 for the ensemble axis
    below = np.expand_dims(below, ensemble_aixs)
    
    # Merge below and f so that it is easier for later processing
    # Now, along the ensemble axis (first axis), the first value
    # is the threshold, and the rest of the values are actual members
    #
    f = np.concatenate([below, f], axis=ensemble_aixs)
    f = np.moveaxis(f, ensemble_aixs, 0)
    
    cores = int(os.environ['pyanen_tqdm_workers'])
    
    if cores == 1:
        if os.environ['pyanen_ens_to_prob_method'] == 'kde':
            with tqdm(total=np.prod(np.delete(f.shape, ensemble_aixs)),
                    disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                    leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                    desc='KDE') as pbar:
                
                ret = np.apply_along_axis(ens_to_prob_kde, 0, f, pbar=pbar)
                
        elif os.environ['pyanen_ens_to_prob_method'] == 'moments':
            ret = ens_to_prob_moments(f)
            
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
        
        n_members_plus_one = f.shape[0]
        f_shape = f.shape[1:]
        
        f = f.reshape(n_members_plus_one, -1)
        
        chunksize = 1000 # int(os.environ['pyanen_tqdm_chunksize'])
        parallelize_axis = -1 # int(os.environ['pyanen_tqdm_map_axis'])
        assert parallelize_axis == -1, 'parallelize_axis needs to be -1 for the ens_to_prob operation. Got {}'.format(parallelize_axis)
        
        ret = process_map(transfer_func,
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
    delta = (1-confidence) / 2
    ci = np.quantile(boot_samples_mean, [delta, 1-delta])
    
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
    
    # Move sample axis to the beginning
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

def _reliability_create_split(f_prob, o_binary, nbins, sample_axis=None):
    
    if sample_axis is None:
        sample_axis = list(range(len(o_binary.shape)))

    assert f_prob.shape == o_binary.shape
    assert hasattr(sample_axis, '__len__'), 'sample_axis must be an array-like object!'

    # Move sample axis to the beginning
    to_axis = 0

    for from_axis in np.sort(sample_axis):
        if from_axis != to_axis:
            f_prob = np.moveaxis(f_prob, from_axis, to_axis)
            o_binary = np.moveaxis(o_binary, from_axis, to_axis)
        to_axis += 1

    shape_to_keep = f_prob.shape[to_axis:]
    
    # Flatten the sample dimensions and the kept dimensions separately
    f_prob = f_prob.reshape(-1, *shape_to_keep)
    o_binary = o_binary.reshape(-1, *shape_to_keep)
    
    # Split array for reliability diagram
    bins = np.linspace(0.0, 1.0 + 1e-8, nbins + 1)
    binids = np.digitize(f_prob, bins) - 1
    
    return binids, f_prob, o_binary, bins


def _reliability_agg_no_boot_slice(binids, y_prob, y_true, bins):

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    
    pred_count = pd.value_counts(binids.flatten()).sort_index()
    true_count = pd.value_counts(y_true.astype(int).flatten()).sort_index()

    pred_count.name = 'Samples in each bin'
    true_count.name = 'Samples in each bin'
    
    return prob_pred, prob_true, pred_count, true_count


def _reliability_agg_no_boot(binids, y_prob, y_true, bins):

    if len(y_true.shape) == 1:
        return _reliability_agg_no_boot_slice(binids, y_prob, y_true, bins)

    else:
        # Initialization
        probs_pred, probs_true, pred_counts, true_counts = [], [], [], []
        
        # Reshape data arrays to two-dimensional arrays with [samples, collapsed dimensions to be kept]
        shape_to_keep = y_true.shape[1:]

        binids = binids.reshape(-1, np.prod(shape_to_keep).astype(int))
        y_prob = y_prob.reshape(-1, np.prod(shape_to_keep).astype(int))
        y_true = y_true.reshape(-1, np.prod(shape_to_keep).astype(int))

        # Process each collapsed slice sequentially
        for i in tqdm(range(binids.shape[1]),
                      disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                      leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                      desc='Reliability aggregation'):

            prob_pred, prob_true, pred_count, true_count = _reliability_agg_no_boot_slice(
                binids[:, i], y_prob[:, i], y_true[:, i], bins)
            
            probs_pred.append(prob_pred)
            probs_true.append(prob_true)
            pred_counts.append(pred_count.to_numpy())
            true_counts.append(true_count.to_numpy())

        # Reconstruct the collapsed dimensions
        probs_pred = np.array(probs_pred).reshape(*shape_to_keep, -1)
        probs_true = np.array(probs_true).reshape(*shape_to_keep, -1)
        pred_counts = np.array(pred_counts).reshape(*shape_to_keep, -1)
        true_counts = np.array(true_counts).reshape(*shape_to_keep, -1)

        return probs_pred, probs_true, pred_counts, true_counts


def _reliability_agg_boot_slice(binids, y_prob, y_true, bins,
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

    probs_pred = np.array(probs_pred)
    probs_true = np.array(probs_true)

    pred_counts = pd.value_counts(binids.flatten()).sort_index()
    true_counts = pd.value_counts(y_true.astype(int).flatten()).sort_index()

    pred_counts.name = 'Samples in each bin'
    true_counts.name = 'Samples in each bin'

    return probs_pred, probs_true, pred_counts, true_counts


def _reliability_agg_boot(binids, y_prob, y_true, bins,
                          n_samples=None, repeats=None, confidence=None, skip_nan=None):

    if len(y_true.shape) == 1:
        return _reliability_agg_boot_slice(binids, y_prob, y_true, bins, n_samples, repeats, confidence, skip_nan)

    else:
        # Initialization
        probs_pred, probs_true, pred_counts, true_counts = [], [], [], []
        
        # Reshape data arrays to two-dimensional arrays with [samples, collapsed dimensions to be kept]
        shape_to_keep = y_true.shape[1:]

        binids = binids.reshape(-1, np.prod(shape_to_keep).astype(int))
        y_prob = y_prob.reshape(-1, np.prod(shape_to_keep).astype(int))
        y_true = y_true.reshape(-1, np.prod(shape_to_keep).astype(int))

        # Process each collapsed slice sequentially
        for i in tqdm(range(binids.shape[1]),
                      disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                      leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                      desc='Reliability aggregation'):

            prob_pred, prob_true, pred_count, true_count = _reliability_agg_boot_slice(
                binids[:, i], y_prob[:, i], y_true[:, i], bins, n_samples, repeats, confidence, skip_nan)
            
            probs_pred.append(prob_pred)
            probs_true.append(prob_true)
            pred_counts.append(pred_count.to_numpy())
            true_counts.append(true_count.to_numpy())

        # Reconstruct the collapsed dimensions
        probs_pred = np.array(probs_pred).reshape(*shape_to_keep, -1, 3)
        probs_true = np.array(probs_true).reshape(*shape_to_keep, -1, 3)
        pred_counts = np.array(pred_counts).reshape(*shape_to_keep, -1)
        true_counts = np.array(true_counts).reshape(*shape_to_keep, -1)

        return probs_pred, probs_true, pred_counts, true_counts


def reliability_diagram(f_prob, o_binary, nbins, sample_axis,
                        boot_samples=None, repeats=None,
                        confidence=None, skip_nan=None):
    
    binids, f_prob, o_binary, bins = _reliability_create_split(
        f_prob=f_prob, o_binary=o_binary, nbins=nbins, sample_axis=sample_axis)
    
    if boot_samples is None:
        return _reliability_agg_no_boot(
            binids, f_prob, o_binary, bins)
    else:
        return _reliability_agg_boot(
            binids, f_prob, o_binary, bins,
            boot_samples, repeats, confidence, skip_nan)
    

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

