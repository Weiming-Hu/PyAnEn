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

try:
    import bottleneck as bn
    use_bn = True
except ImportError:
    use_bn = False


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
def wrapper_cdf(x, cdf_func): return cdf_func(below=x)
def wrapper_brier(x, workflow_func, brier_func): return workflow_func('_'.join(['brier', str(None), str(x)]), brier_func, over=None, below=x)


def integrate(verifier, type, integration_range=None, nbins=20):
    
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
        wrapper = partial(wrapper_brier, workflow_func=verifier._metric_workflow_1, brier_func=verifier._brier)
        
        if cores == 1:
            brier = np.array([wrapper(_x) for _x in tqdm(seq_x, **pbar_kws)])
        else:
            brier = np.array(process_map(wrapper, seq_x, max_workers=cores, chunksize=chunksize, **pbar_kws))
        
        # Calculate difference
        dx = (seq_x[1:] - seq_x[:-1]).reshape(nbins - 1, *(len(brier.shape[1:]) * [1]))
        
        # Integrate
        if use_bn:
            ret = 0.5 * bn.nansum((brier[1:] + brier[:-1]) * dx, axis=0)
        else:
            ret = 0.5 * np.nansum((brier[1:] + brier[:-1]) * dx, axis=0)
        
    elif type == 'mean':
        # Reference: https://math.berkeley.edu/~scanlon/m16bs04/ln/16b2lec30.pdf
        
        # Calculate CDF at bins
        wrapper = partial(wrapper_cdf, cdf_func=verifier.cdf)
        
        if cores == 1:
            cdf = np.array([wrapper(_x) for _x in tqdm(seq_x, **pbar_kws)])
        else:
            cdf = np.array(process_map(wrapper, seq_x, max_workers=cores, chunksize=chunksize, **pbar_kws))
        
        # Calculate difference
        d_cdf = cdf[1:] - cdf[:-1]
        x = seq_x.reshape(nbins, *(len(cdf.shape[1:]) * [1]))
        
        # Integrate
        if use_bn:
            ret = 0.5 * bn.nansum((x[1:] + x[:-1]) * d_cdf, axis=0)
        else:
            ret = 0.5 * np.nansum((x[1:] + x[:-1]) * d_cdf, axis=0)
    
    elif type == 'variance':
        # Reference: https://math.berkeley.edu/~scanlon/m16bs04/ln/16b2lec30.pdf
        
        # Calculate CDF at bins
        wrapper = partial(wrapper_cdf, verifier=verifier)
        
        if cores == 1:
            cdf = np.array([wrapper(_x) for _x in tqdm(seq_x, **pbar_kws)])
        else:
            cdf = np.array(process_map(wrapper, seq_x, max_workers=cores, chunksize=chunksize, **pbar_kws))
        
        # Calculate difference
        d_cdf = cdf[1:] - cdf[:-1]
        x = seq_x.reshape(nbins, *(len(cdf.shape[1:]) * [1]))
        x_sq = x ** 2
        
        # Integrate
        if use_bn:
            mean = 0.5 * bn.nansum((x[1:] + x[:-1]) * d_cdf, axis=0)
            ret = 0.5 * bn.nansum((x_sq[1:] + x_sq[:-1]) * d_cdf, axis=0) - mean ** 2
        else:
            mean = 0.5 * np.nansum((x[1:] + x[:-1]) * d_cdf, axis=0)
            ret = 0.5 * np.nansum((x_sq[1:] + x_sq[:-1]) * d_cdf, axis=0) - mean ** 2

    else:
        raise Exception('Unknon type of integration. Got {}'.format(type))
    
    # Resume sub-level progress bars
    os.environ['pyanen_tqdm_disable'] = str(pbar_kws['disable'])
    os.environ['pyanen_tqdm_workers'] = str(cores)
    
    return ret
    