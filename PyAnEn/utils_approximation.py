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
from functools import partial
from tqdm.contrib.concurrent import process_map


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


# Wrapper functions for parallelization
def wrapper_cdf(x, verifier): return verifier.cdf(below=x)
def wrapper_brier(x, verifier): return verifier._metric_workflow_1('_'.join(['brier', str(None), str(x)]), verifier._brier, over=None, below=x)


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
    
    cores = int(os.environ['pyanen_tqdm_workers'])
    chunksize = int(os.environ['pyanen_tqdm_chunksize'])
    
    # Suppress sub-level progress bars and parallelization
    os.environ['pyanen_tqdm_disable'] = 'True'
    
    if cores > 1:
        os.environ['pyanen_tqdm_workers'] = '1'
    
    if type == 'brier':
        
        # Calculate briers at bins
        wrapper = partial(wrapper_brier, verifier=verifier)
        
        if cores == 1:
            seq_y = np.array([wrapper(_x) for _x in tqdm(seq_x, **pbar_kws)])
        else:
            seq_y = np.array(process_map(wrapper, seq_x, max_workers=cores, chunksize=chunksize, **pbar_kws))
        
        # Calculate difference
        seq_x = seq_x[1:] - seq_x[:-1]
        seq_x = seq_x.reshape(nbins - 1, *(len(seq_y.shape[1:]) * [1]))
        
        # Integrate
        ret = 0.5 * np.nansum((seq_y[1:] + seq_y[:-1]) * seq_x, axis=0)
        
    elif type == 'cdf':
        
        # Calculate CDF at bins
        wrapper = partial(wrapper_cdf, verifier=verifier)
        
        if cores == 1:
            seq_y = np.array([wrapper(_x) for _x in tqdm(seq_x, **pbar_kws)])
        else:
            seq_y = np.array(process_map(wrapper, seq_x, max_workers=cores, chunksize=chunksize, **pbar_kws))
        
        # Calculate difference
        seq_y = seq_y[1:] - seq_y[:-1]
        seq_x = seq_x[1:].reshape(nbins - 1, *(len(seq_y.shape[1:]) * [1]))
        
        # Integrate
        ret = np.nansum(seq_x * seq_y, axis=0)

    else:
        raise Exception('Unknon type of integration. Got {}'.format(type))
    
    # Resume sub-level progress bars
    os.environ['pyanen_tqdm_disable'] = str(pbar_kws['disable'])
    os.environ['pyanen_tqdm_workers'] = str(cores)
    
    return (seq_x, seq_y) if return_slices else ret
    