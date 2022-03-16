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
# Date of Creation: 2022/03/15                              #
#############################################################
#
# Utility functions for CRPS
#

import os

import numpy as np

from distutils import util
from scipy import stats, special


def _lbeta(x1, x2):
    log_prod_gamma_x = np.log(special.gamma(x1)) + np.log(special.gamma(x2))
    log_gamma_sum_x = np.log(special.gamma(x1 + x2))
    return log_prod_gamma_x - log_gamma_sum_x


def crps_gamma(mu, sigma, shift, obs, reduce_sum=True):
    # Code referenced from Mohammadvaghef Ghazvinian
    # Based on the following paper:
    # https://doi.org/10.1016/j.advwatres.2021.103907
    
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


def crps_truncated_gaussian(y, mu=0, scale=1, l=0):
    # Reference:
    # 1. https://cran.microsoft.com/snapshot/2017-09-17/web/packages/scoringRules/vignettes/crpsformulas.html#GenNormal
    # 2. https://github.com/cran/scoringRules/blob/f7b7df5f1ba32247be92e98095afff2d8fee9eb6/R/scores_norm.R#L190
    #
    # Assume u = +inf, U = 0 and L = 0

    y = y - mu
    l = l - mu
    
    l = l / scale
    y = y / scale
    
    z = np.copy(y)
    mask = z < l
    z[mask] = l[mask]
    
    dist = stats.norm()
    f_z = dist.pdf(z)
    F_z = dist.cdf(z)
    F_l = dist.cdf(l)
    F_u = 1
    
    F_usqrt2 = 1
    F_lsqrt2 = dist.cdf(l * np.sqrt(2))
    
    # Calculate intermediate terms
    frac1 = 1 / (F_u - F_l)
    
    # Calculate separate terms
    term1 = np.abs(y - z)
    term2 = frac1 * z * (2 * F_z - F_u - F_l)
    term3 = frac1 * 2 * f_z
    term4 = frac1 ** 2 * 1 / np.sqrt(np.pi) * (F_usqrt2 - F_lsqrt2)
    
    # Calculate final metric
    crps = term1 + term2 + term3 - term4
    
    return crps * scale
