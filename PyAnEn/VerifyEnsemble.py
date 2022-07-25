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
# Class definition for ensemble forecast verification
#

import os
import numpy as np
import properscoring as ps

from distutils import util
from tqdm.auto import tqdm
from .Verify import Verify
from .utils_special_boot import corr
from .utils_special_boot import brier_decomp
from .utils_verify import iou_prob
from .utils_verify import iou_determ
from .utils_verify import ens_to_prob
from .utils_verify import binarize_obs
from .utils_verify import calculate_roc
from .utils_verify import rank_histogram
from .utils_verify import _reliability_create_split
from .utils_verify import _binned_spread_skill_create_split



class VerifyEnsemble(Verify):
        
    def __init__(self, f, o, ensemble_axis=None, ensemble_collapse_func=np.mean,
                 avg_axis=None, boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        self.f_collapsed = None
        self.ensemble_axis = ensemble_axis
        self.ensemble_collapse_func = ensemble_collapse_func
        
        super().__init__(avg_axis=avg_axis, boot_samples=boot_samples,
                         working_directory=working_directory,
                         start_from_scratch=start_from_scratch)
        
        # Calculate the deterministic form of the ensemble forecasts
        self._collapse_ensembles()
    
    def set_ensemble_axis(self, axis):
        self.ensemble_axis = axis
        self._validate()
        return self
    
    def set_collapse_func(self, func):
        self.ensemble_collapse_func = func
        self._validate()
        return self
    
    ###################
    # Private Methods #
    ###################
    
    ###### Metric Methods ######
    
    def _f_determ(self):
        return np.copy(self.f_collapsed)
    
    def _corr(self):
        return corr(self.f_determ(), self.o, self.avg_axis, self.boot_samples)
    
    def _error(self):
        return self.f_collapsed - self.o
    
    def _sq_error(self):
        return (self.f_collapsed - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self.f_collapsed - self.o)
    
    def _crps(self):
        if util.strtobool(os.environ['pyanen_split_crps_ensemble_along_0']):
            
            # Move ensemble axis to the last one if it is not already
            if self.ensemble_axis == -1 or self.ensemble_axis == len(self.f.shape) - 1:
                _f = self.f
            else:
                _f = np.moveaxis(self.f, self.ensemble_axis, -1)
                
            crps = [ps.crps_ensemble(self.o[i], _f[i], axis=-1)
                    for i in tqdm(
                        range(self.o.shape[0]),
                        disable=util.strtobool(os.environ['pyanen_tqdm_disable']),
                        leave=util.strtobool(os.environ['pyanen_tqdm_leave']),
                        desc='Ensemble CRPS (along first axis)')]
            
            return np.stack(crps, axis=0)
        
        else:
            return ps.crps_ensemble(observations=self.o, forecasts=self.f, axis=self.ensemble_axis)
    
    def _rank_hist(self):
        return rank_histogram(f=self.f, o=self.o, ensemble_axis=self.ensemble_axis)
    
    def _spread(self):
        if util.strtobool(os.environ['pyanen_skip_nan']):
            return np.nanmax(self.f, axis=self.ensemble_axis) - np.nanmin(self.f, axis=self.ensemble_axis)
        else:
            return np.max(self.f, axis=self.ensemble_axis) - np.min(self.f, axis=self.ensemble_axis)
    
    def _brier(self, over, below):
        brier = self.cdf(over=over, below=below) - binarize_obs(self.o, over=over, below=below)
        return brier ** 2
    
    def _brier_decomp(self, over, below):
        f = self.cdf(over=over, below=below)
        o = binarize_obs(self.o, over=over, below=below)
        return brier_decomp(f, o, self.avg_axis, self.boot_samples)
    
    def _variance(self):
        return self.f.var(self.ensemble_axis)
    
    def _binned_spread_skill(self, nbins=15):
        
        # Calculate variances and squared errors
        # Not using the public calls (self.variance and self.sq_error) because no aggregation is needed!
        #
        variance = self._metric_workflow_1('variance', self._variance)
        sq_error = self._metric_workflow_1('sq_error', self._sq_error)
        
        return _binned_spread_skill_create_split(variance, sq_error, nbins=nbins, sample_axis=self.avg_axis)
    
    def _reliability(self, nbins=15, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self.cdf(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return _reliability_create_split(f_prob, o_binary, nbins)
    
    def _roc(self, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self.cdf(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_roc(f_prob, o_binary)
    
    def _sharpness(self, over=None, below=None):
        return self.cdf(over=over, below=below)
    
    def _iou_determ(self, over=None, below=None):
        return iou_determ(self.f_collapsed, self.o, axis=self.avg_axis, over=over, below=below)
    
    def _iou_prob(self, over=(None, None), below=(None, None)):
        
        assert isinstance(over, tuple) or isinstance(over, list)
        assert isinstance(below, tuple) or isinstance(below, list)
        assert len(over) == 2 or len(below) == 2
        
        f_prob = self.cdf(over=over[0], below=below[0])
        o_binary = binarize_obs(self.o, over=over[0], below=below[0])
        
        return iou_prob(f_prob, o_binary, axis=self.avg_axis, over=over[1], below=below[1])
    
    def _cdf(self, over=None, below=None):
        assert (over is None) ^ (below is None), 'Must specify over or below'
        if below is None: return 1 - self.cdf(below=over)
        else: return ens_to_prob(self.f, ensemble_aixs=self.ensemble_axis, over=over, below=below)
    
    ###### Other Methods ######
    
    def _guess_ensemble_axis(self):
        
        # I will try to guess either the last or the first dimension is the ensemble member axis
        my_guesses = [-1, 0]
        o_shape = list(self.o.shape)
        
        for guess in my_guesses:
            f_shape = list(self.f.shape)
            f_shape.pop(guess)
            
            if f_shape == o_shape:
                self.ensemble_axis = guess
                return
                
        raise Exception('Specify ensemble_axis! Got f {} and o {}'.format(self.f.shape, self.o.shape))
    
    def _validate(self):
        super()._validate()
        
        # Check forecasts and observations
        assert isinstance(self.f, np.ndarray)
        assert isinstance(self.o, np.ndarray)
        
        # Check ensemble axis
        if self.ensemble_axis is None:
            self._guess_ensemble_axis()
        else:
            assert isinstance(self.ensemble_axis, int)
            
        assert self.ensemble_axis == 0 or self.ensemble_axis == -1 or \
            self.ensemble_axis == -len(self.f.shape) or \
            self.ensemble_axis == len(self.f.shape) - 1, \
            'Ensemble axis needs to be either the first or the last'
        
        # Check dimensions
        f_shape, o_shape = list(self.f.shape), list(self.o.shape)
        f_shape.pop(self.ensemble_axis)
        assert f_shape == o_shape, 'Shape mismatch: f ({}) and o ({})'.format(f_shape, o_shape)
        
        # Check collapse function
        assert callable(self.ensemble_collapse_func)
    
    def _collapse_ensembles(self):
        self.f_collapsed = self.ensemble_collapse_func(self.f, axis=self.ensemble_axis)
    
    def __repr__(self):
        msg = super().__repr__()
        msg += '\nEnsemble axis (ensemble_axis): {}'.format(self.ensemble_axis)
        msg += '\nEnsemble collapsing function (ensemble_collapse_func): {}'.format(self.ensemble_collapse_func.__name__)
        return msg
