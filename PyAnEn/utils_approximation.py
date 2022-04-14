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


# Wrapper functions for parallelizing CDF
def wrapper_cdf(x, verifier, memmap_arr=None):
    if memmap_arr is None: return verifier.cdf(below=x)
    else: memmap_arr[x[0]] = verifier.cdf(below=x[1])

# Wrapper function for parallelizing Brier.
# Not using the public function call because brier scores need not to be aggregated.
# Intermediately saved results can still be used because _brier will call cdf that relies on saved data.
#
def wrapper_brier(x, verifier, memmap_arr=None):
    if memmap_arr is None: return verifier._metric_workflow_1(verifier.to_name('brier', over=None, below=x), verifier._brier, over=None, below=x)
    else: memmap_arr[x[0]] = verifier._metric_workflow_1(verifier.to_name('brier', over=None, below=x[1]), verifier._brier, over=None, below=x[1])


class Integration:
    def __init__(self, verifier, integration_range=None, nbins=20,
                 disable_pbar=None, leave_pbar=None, pbar_desc=None,
                 workers=None, chunksize=None, less_memory=None,
                 memmap_dir=None, memmap_dtype=np.float64):
        
        # Initialization
        self.seq_x = None
        self.nbins = nbins
        self.tqdm_backup = None
        self.verifier = verifier
        self.integration_range = integration_range
        self.memmap_dtype = memmap_dtype
        
        # Define the range for numerical integration
        self._define_integration_range()
        
        # Define progress bar parameters
        self.pbar_kws = {
            'disable': util.strtobool(os.environ['pyanen_tqdm_disable']) if disable_pbar is None else disable_pbar,
            'leave': util.strtobool(os.environ['pyanen_tqdm_leave']) if leave_pbar is None else leave_pbar,
        }
        
        self.cores = int(os.environ['pyanen_tqdm_workers']) if workers is None else workers
        self.chunksize = int(os.environ['pyanen_tqdm_chunksize']) if chunksize is None else chunksize
        self.less_memory = util.strtobool(os.environ['pyanen_integrate_with_less_memory']) if less_memory is None else less_memory
        
        wdir = os.environ['pyanen_integrate_wdir']
        
        if memmap_dir is not None: self.memmap_dir = memmap_dir
        elif wdir != '': self.memmap_dir = os.path.expanduser(wdir)
        elif self.verifier.working_directory is not None: self.memmap_dir = self.verifier.working_directory
        else: self.memmap_dir = ''
        
    def crps(self): return self._workflow(self._crps)
    def mean(self): return self._workflow(self._mean)
    def variance(self): return self._workflow(self._variance)
        
    ###################
    # Private Methods #
    ###################
        
    def _crps(self):
        desc = 'Integrating CRPS for ' + type(self.verifier).__name__
        wrapper = partial(wrapper_brier, verifier=self.verifier, memmap_arr=self.memmap_arr_w)
        iterables = enumerate(self.seq_x) if self.less_memory else self.seq_x
        
        if self.cores == 1: brier = np.array([wrapper(_x) for _x in tqdm(iterables, **self.pbar_kws, desc=desc, total=self.nbins)])
        else: brier = np.array(process_map(wrapper, iterables, max_workers=self.cores, chunksize=self.chunksize, **self.pbar_kws, desc=desc, total=self.nbins))
        
        # Calculate difference
        dx = (self.seq_x[1:] - self.seq_x[:-1]).reshape(self.nbins - 1, *(len(self.verifier.o.shape) * [1]))
        
        if self.less_memory:
            self.memmap_arr_w.flush()
            return 0.5 * (self._memmap_sum_mul(self.memmap_arr_r[1:], dx, self.memmap_dtype) +
                          self._memmap_sum_mul(self.memmap_arr_r[:-1], dx, self.memmap_dtype))
        else:
            return 0.5 * np.sum((brier[1:] + brier[:-1]) * dx, axis=0)
        
    def _mean(self):
        # Reference: https://math.berkeley.edu/~scanlon/m16bs04/ln/16b2lec30.pdf
        
        desc = 'Integrating mean for ' + type(self.verifier).__name__
        wrapper = partial(wrapper_cdf, verifier=self.verifier, memmap_arr=self.memmap_arr_w)
        iterables = enumerate(self.seq_x) if self.less_memory else self.seq_x
        
        if self.cores == 1: cdf = np.array([wrapper(_x) for _x in tqdm(iterables, **self.pbar_kws, desc=desc, total=self.nbins)])
        else: cdf = np.array(process_map(wrapper, iterables, max_workers=self.cores, chunksize=self.chunksize, **self.pbar_kws, desc=desc, total=self.nbins))
        
        # Calculate difference
        x = self.seq_x.reshape(self.nbins, *(len(self.verifier.o.shape) * [1]))
        x2 = x[1:] + x[:-1]
        
        if self.less_memory:
            self.memmap_arr_w.flush()
            return 0.5 * (self._memmap_sum_mul(self.memmap_arr_r[1:], x2, self.memmap_dtype) -
                          self._memmap_sum_mul(self.memmap_arr_r[:-1], x2, self.memmap_dtype))
        else:        
            return 0.5 * np.sum(x2 * (cdf[1:] - cdf[:-1]), axis=0)
    
    def _variance(self):
        # Reference: https://math.berkeley.edu/~scanlon/m16bs04/ln/16b2lec30.pdf
        
        desc = 'Integrating variance ' + type(self.verifier).__name__
        wrapper = partial(wrapper_cdf, verifier=self.verifier, memmap_arr=self.memmap_arr_w)
        iterables = enumerate(self.seq_x) if self.less_memory else self.seq_x
        
        if self.cores == 1: cdf = np.array([wrapper(_x) for _x in tqdm(iterables, **self.pbar_kws, desc=desc, total=self.nbins)])
        else: cdf = np.array(process_map(wrapper, iterables, max_workers=self.cores, chunksize=self.chunksize, **self.pbar_kws, desc=desc, total=self.nbins))
        
        # Calculate difference
        x = self.seq_x.reshape(self.nbins, *(len(self.verifier.o.shape) * [1]))
        x2 = x[1:] + x[:-1]
        
        x_sq = x ** 2
        x_sq2 = x_sq[1:] + x_sq[:-1]
        
        if self.less_memory:
            self.memmap_arr_w.flush()
            mean = 0.5 * (self._memmap_sum_mul(self.memmap_arr_r[1:], x2, self.memmap_dtype) -
                          self._memmap_sum_mul(self.memmap_arr_r[:-1], x2, self.memmap_dtype))
            var = 0.5 * (self._memmap_sum_mul(self.memmap_arr_r[1:], x_sq2, self.memmap_dtype) -
                         self._memmap_sum_mul(self.memmap_arr_r[:-1], x_sq2, self.memmap_dtype)) - mean ** 2
            
        else:
            mean = 0.5 * np.sum(x2 * (cdf[1:] - cdf[:-1]), axis=0)
            var = 0.5 * np.sum(x_sq2 * (cdf[1:] - cdf[:-1]), axis=0) - mean ** 2
            
        return var
    
    def _workflow(self, func):
        
        self._suppress_sublevel_tqdm()
        self._initialize_memmap(memmap_shape=tuple([self.nbins] + list(self.verifier.o.shape)))
        ret = func()
        self._finalize_memmap()
        self._resume_sublevel_tqdm()
        
        return ret
    
    def _initialize_memmap(self, memmap_shape):
        self.memmap_arr_w = np.memmap(
            filename=os.path.join(self.memmap_dir, '__numerical_integration__.dat'),
            shape=memmap_shape, dtype=self.memmap_dtype, mode='write') if self.less_memory else None
        
        self.memmap_arr_r = np.memmap(
            filename=os.path.join(self.memmap_dir, '__numerical_integration__.dat'),
            shape=memmap_shape, dtype=self.memmap_dtype, mode='readonly') if self.less_memory else None
    
    def _finalize_memmap(self):
        if self.less_memory:
            del self.memmap_arr_w
            del self.memmap_arr_r
            os.remove(os.path.join(self.memmap_dir, '__numerical_integration__.dat'))
    
    def _suppress_sublevel_tqdm(self):
        
        # Backup environment variables
        self.tqdm_backup = {
            'pyanen_tqdm_disable': util.strtobool(os.environ['pyanen_tqdm_disable']),
            'pyanen_tqdm_workers': int(os.environ['pyanen_tqdm_workers'])
        }
        
        # Disable sublevel prograss bar to prevent an too many bars showing up
        os.environ['pyanen_tqdm_disable'] = 'True'
        
        # Disable sublevel parallelization to ensure better performance
        if self.cores > 1:
            os.environ['pyanen_tqdm_workers'] = '1'
    
    def _resume_sublevel_tqdm(self):
        assert self.tqdm_backup is not None, 'Resume function called before calling the suppressing function!'
        
        for k, v in self.tqdm_backup.items():
            os.environ[k] = str(v)
        
    def _define_integration_range(self):
        
        if self.integration_range is None:
        
            range_multi = int(os.environ['pyanen_integrate_range_multiplier'])
            assert range_multi > 0
            
            # Define the x bins to integrate
            vmin = np.min(self.verifier.o)
            vmax = np.max(self.verifier.o)
            
            vrange = vmax - vmin
            self.integration_range = (vmin - range_multi * vrange, vmax + range_multi * vrange)
            
        self.seq_x = np.linspace(self.integration_range[0], self.integration_range[1], self.nbins)
    
    def _memmap_sum_mul(self, arr, mul, dtype):
        
        assert arr.shape[0] == mul.shape[0], 'Assert {} == {}'.format(arr.shape[0], mul.shape[0])
        assert len(arr.shape) == len(mul.shape), 'Assert {} == {}'.format(len(arr.shape), len(mul.shape))
    
        out = np.full(shape=arr.shape[1:], fill_value=0, dtype=dtype)
        
        for i in tqdm(range(arr.shape[0]), desc='Aggregating memmap', **self.pbar_kws, total=arr.shape[0]):
            out += arr[i] * mul[i]
            
        return out
