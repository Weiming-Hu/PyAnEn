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
# Date of Creation: 2022/03/22                              #
#############################################################
#
# Utility functions for approximating statistics
#

import os

import numpy as np

from tqdm.auto import tqdm
from distutils import util


# Additional helper functions
def _get_integration_range(verifier, nbins):
    range_multi = int(os.environ['pyanen_integrate_range_multiplier'])
    assert range_multi > 0
    
    # Define the x bins to integrate
    vmin = np.nanmin(verifier.o)
    vmax = np.nanmax(verifier.o)
    
    vrange = vmax - vmin
    
    integration_min = vmin - range_multi * vrange
    integration_max = vmax + range_multi * vrange
    
    return np.linspace(integration_min, integration_max, nbins)


def integrate(verifier, type, integration_range=None, nbins=20, return_slices=False):
    
    assert nbins > 5, 'Too few bins to integrate! Got {}'.format(nbins)
    
    if integration_range is None:
        seq_x = _get_integration_range(verifier, nbins)
    else:
        seq_x = np.linspace(integration_range[0], integration_range[1], nbins)
    
    pbar_kws = {
        'disable': util.strtobool(os.environ['pyanen_tqdm_disable']),
        'leave': util.strtobool(os.environ['pyanen_tqdm_leave']),
        'desc': 'Integrating {}'.format(type),
    }
    
    if type == 'brier':
        # Calculate briers at bins
        seq_brier = np.array([verifier.brier(below=_x) for _x in tqdm(seq_x, **pbar_kws)])
        
        # Calculate difference
        diff_x = seq_x[1:] - seq_x[:-1]
        
        # Integrate
        crps = 0.5 * np.sum((seq_brier[1:] + seq_brier[:-1]) * diff_x)
        
        return (diff_x, seq_brier) if return_slices else crps
        
    elif type == 'cdf':
        
        # Calculate CDF at bins
        seq_cdf = np.array([verifier.cdf(below=_x) for _x in tqdm(seq_x, **pbar_kws)])
        
        # Calculate difference
        diff_cdf = seq_cdf[1:] - seq_cdf[:-1]
        seq_x = seq_x[1:].reshape(nbins - 1, *(len(diff_cdf.shape[1:]) * [1]))    
        
        # Integrate
        cdf = np.nansum(seq_x * diff_cdf, axis=0)
        
        return (seq_x, diff_cdf) if return_slices else cdf

    else:
        raise Exception('Unknon type of integration. Got {}'.format(type))
    