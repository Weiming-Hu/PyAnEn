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

import numpy as np
import properscoring as ps

from functools import partial

from .Verify import Verify
from .utils_verify import ens_to_prob
from .utils_verify import binarize_obs
from .utils_verify import calculate_roc
from .utils_verify import rank_histogram
from .utils_verify import calculate_reliability
from .utils_verify import _binned_spread_skill_create_split


class VerifyEnsemble(Verify):
        
    def __init__(self, f, o, ensemble_axis=None, ensemble_collapse_func=np.mean,
                 avg_axis=None, boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        self.f_determ = None
        self.ensemble_axis = ensemble_axis
        self.ensemble_collapse_func = ensemble_collapse_func
        
        super().__init__(avg_axis=avg_axis, boot_samples=boot_samples,
                         working_directory=working_directory,
                         start_from_scratch=start_from_scratch)
        
        self._collapse_ensembles()
    
    ###################
    # Private Methods #
    ###################
    
    ###### Metric Methods ######
    
    def _error(self):
        return self.f_determ - self.o
    
    def _sq_error(self):
        return (self.f_determ - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self.f_determ - self.o)
    
    def _crps(self):
        return ps.crps_ensemble(observations=self.o, forecasts=self.f, axis=self.ensemble_axis)
    
    def _rank_hist(self):
        return rank_histogram(f=self.f, o=self.o, ensemble_axis=self.ensemble_axis)
    
    def _spread(self):
        return np.max(self.f, axis=self.ensemble_axis) - np.min(self.f, axis=self.ensemble_axis)
    
    def _brier(self, over=None, below=None):
        brier = self._ens_to_prob(over=over, below=below) - binarize_obs(self.o, over=over, below=below)
        return brier ** 2
    
    def _variance(self):
        return self.f.var(self.ensemble_axis)
    
    def _binned_spread_skill(self, nbins=15):
        
        # Calculate variances and squared errors
        variance = self._metric_workflow_1('variance', self._variance)
        ab_error = self._metric_workflow_1('ab_error', self._ab_error)
        
        return _binned_spread_skill_create_split(variance, ab_error, nbins=nbins, sample_axis=self.avg_axis)
    
    def _reliability(self, nbins=15, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self._ens_to_prob(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_reliability(f_prob, o_binary, nbins)
    
    def _roc(self, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self._ens_to_prob(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_roc(f_prob, o_binary)
    
    def _sharpness(self, over=None, below=None):
        return self._ens_to_prob(over=over, below=below)
    
    ###### Other Methods ######
    
    def _ens_to_prob(self, over=None, below=None):
        f_prob_func = partial(ens_to_prob, f=self.f, ensemble_aixs=self.ensemble_axis, over=over, below=below)
        return self._metric_workflow_1('_'.join(['ens_to_prob', str(over), str(below)]), f_prob_func)
    
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
        
        # Check dimensions
        f_shape, o_shape = list(self.f.shape), list(self.o.shape)
        f_shape.pop(self.ensemble_axis)
        assert f_shape == o_shape, 'Shape mismatch: f ({}) and o ({})'.format(f_shape, o_shape)
        
        # Check collapse function
        assert callable(self.ensemble_collapse_func)
    
    def _collapse_ensembles(self):
        self.f_determ = self.ensemble_collapse_func(self.f, axis=self.ensemble_axis)
    
    def __str__(self):
        msg = super().__str__()
        msg += '\nForecasts (f): {}'.format(self.f.shape)
        msg += '\nObservations (o): {}'.format(self.o.shape)
        msg += '\nEnsemble axis (ensemble_axis): {}'.format(self.ensemble_axis)
        msg += '\nEnsemble collapsing function (ensemble_collapse_func): {}'.format(self.ensemble_collapse_func.__name__)
        return msg
