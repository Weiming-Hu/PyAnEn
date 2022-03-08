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

import numpy as np

import scipy.stats as st
from sklearn import metrics
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity


DEFAULTS = {
    'kde_bandwidth': 0.01,
    'kde_kernel': 'gaussian',
    'kde_samples': 20,
    'boot_confidence': 0.95,
    'boot_repeats': 1000,
    'boot_samples': 3000,
}


def rank_histogram(f, o, ensemble_axis):
    # Reference:
    # https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
    
    # Move the ensemble axis to the first axis and
    # combine observations and ensembles along the first axis
    #
    c = np.vstack((
        np.expand_dims(o, 0),
        np.moveaxis(f, ensemble_axis, 0)))
    
    # Calculate ranks for ensemble members
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'), 0, c)
    
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

    return obs_ranks


def ens_to_prob_kde(ens, over=None, below=None,
                    bandwidth=DEFAULTS['kde_bandwidth'],
                    kernel=DEFAULTS['kde_kernel'],
                    samples=DEFAULTS['kde_samples']):
    
    ens = ens.reshape(-1, 1)
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(ens)
    
    vmin, vmax = ens.mean(), ens.max()
    spread = vmax - vmin
    
    x = np.linspace(vmin - spread, below, samples) \
        if over is None else np.linspace(over, vmax + spread, samples)
        
    return np.sum(np.exp(kde.score_samples(x)))


def ens_to_prob(f, ensemble_aixs, over=None, below=None):
    assert (over is None) ^ (below is None), 'Must specify over or below'
    return np.apply_along_axis(ens_to_prob_kde, ensemble_aixs, f)
    

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
    arr_split = np.array_split(np.stack([f_prob, o_binary], axis=1), nbins, 0)
    
    # Calculate average forecasted probability and the average observed frequency
    rel = np.array([np.mean(i, axis=0) for i in arr_split])
    forecasted, observed = rel[:, 0], rel[:, 1]
    
    return forecasted, observed


def calculate_roc(f_prob, o_binary):
    # Flatten
    f_prob = f_prob.flatten()
    o_binary = o_binary.flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=o_binary, y_score=f_prob)
    auc = metrics.auc(fpr, tpr)
    
    return fpr, tpr, auc


##############################
# Functions for Bootstraping #
##############################

def boot_vec(pop, n_samples=DEFAULTS['boot_samples'], repeats=DEFAULTS['boot_repeats'], confidence=DEFAULTS['boot_confidence']):
    boot_samples_mean = [np.random.choice(pop, size=n_samples, replace=True).mean() for _ in range(repeats)]
    boot_samples_mean = np.array(boot_samples_mean)

    sample_mean = boot_samples_mean.mean()
    sample_std = boot_samples_mean.std()

    ci = st.t.interval(alpha=confidence, df=repeats-1, loc=sample_mean, scale=sample_std)
    return np.array([ci[0], sample_mean, ci[1]])


def boot_arr(metric, sample_axis, n_samples=DEFAULTS['boot_samples'], repeats=DEFAULTS['boot_repeats'], confidence=DEFAULTS['boot_confidence']):
    
    # Move average axis to the beginning
    to_axis = 0

    for from_axis in np.sort(sample_axis):
        if from_axis != to_axis:
            metric = np.moveaxis(metric, from_axis, to_axis)
        to_axis += 1

    shape_to_keep = metric.shape[to_axis:]

    # Flatten the sample dimensions and the kept dimensions separately
    metric = metric.reshape(-1, *shape_to_keep)
    metric = metric.reshape(-1, np.prod(shape_to_keep))
    
    intervals = np.apply_along_axis(boot_vec, 0, metric, n_samples=n_samples, repeats=repeats, confidence=confidence)
    intervals = intervals.reshape(-1, *shape_to_keep)
    
    return intervals


##########################################
# Functions for Spread Skill Correlation #
##########################################

def _binned_spread_skill_create_split(variance, ab_error, nbins, sample_axis):
    assert variance.shape == ab_error.shape
    
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
    
    variance = variance.reshape(-1, np.prod(shape_to_keep))
    ab_error = ab_error.reshape(-1, np.prod(shape_to_keep))
    
    # Sort arrays based on error
    for i in range(variance.shape[1]):
        index_sort = np.argsort(variance[:, i])
        variance[:, i] = variance[index_sort, i]
        ab_error[:, i] = ab_error[index_sort, i]
    
    # Combine and split arrays
    # The first column is absolute error
    # The second column is standard deviation
    #
    arr_split = np.array_split(np.stack([ab_error, np.sqrt(variance)], axis=1), nbins, 0)
    
    return arr_split, shape_to_keep

def _binned_spread_skill_agg_no_boot(arr_split, reconstruct_shape):
    corrs = []
    
    for arr in arr_split:
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        corr = np.array([np.corrcoef(arr[:, 0, i], arr[:, 1, i])[0, 1] for i in range(arr.shape[2])])
        corrs.append(corr)
    
    corrs = np.stack(corrs, axis=0)
    assert corrs.shape[1] == np.prod(reconstruct_shape)
    
    corrs = corrs.reshape(corrs.shape[0], *reconstruct_shape)
    return corrs


def _binned_spread_skill_agg_boot(arr_split, reconstruct_shape,
                                  n_samples=DEFAULTS['boot_samples'],
                                  repeats=DEFAULTS['boot_repeats'],
                                  confidence=DEFAULTS['boot_confidence']):
    corrs = []
    
    for arr in arr_split:
        assert len(arr.shape) == 3 and arr.shape[1] == 2
        pop_size = arr.shape[0]
        
        corrs_ci = []
        
        for dim_i in range(arr.shape[2]):
            corrs = []
            
            for _ in range(repeats):
                idx = np.random.randint(pop_size, size=n_samples)
                corr = np.corrcoef(arr[idx, 0, dim_i], arr[idx, 1, dim_i])[0, 1]
                corrs.append(corr)
            
            corrs = np.array(corrs)
            sample_mean = corrs.mean()
            sample_std = corrs.std()

            ci = st.t.interval(alpha=confidence, df=repeats-1, loc=sample_mean, scale=sample_std)
            corrs_ci.append([ci[0], sample_mean, ci[1]])
        
        corrs_ci = np.stack(corrs_ci, axis=0)
        assert len(corrs_ci.shape) == 2 and corrs_ci.shape[1] == np.prod(reconstruct_shape)
        
        corrs_ci = corrs_ci.reshape(3, *reconstruct_shape)
        return corrs_ci


def bin_spread_skill(
    variance, ab_error, nbins, sample_axis,
    boot_samples=None, repeats=DEFAULTS['boot_repeats'],
    confidence=DEFAULTS['boot_confidence']):
    
    arr_split, shape_to_keep = _binned_spread_skill_create_split(
        variance=variance, ab_error=ab_error,
        nbins=nbins, sample_axis=sample_axis)
    
    if boot_samples is None:
        return _binned_spread_skill_agg_no_boot(
            arr_split=arr_split, reconstruct_shape=shape_to_keep)
    else:
        return _binned_spread_skill_agg_boot(
            arr_split=arr_split, reconstruct_shape=shape_to_keep,
            n_samples=boot_samples, repeats=repeats, confidence=confidence)
        
        
if __name__ == '__main__':
    f = np.arange(0, 40).reshape(2, 4, 5)
    o = np.array([0.1, 7.7, 9, 20, 22, 27, 32, 37]).reshape(2, 4)
    r = rank_histogram(f, o, 2)
    