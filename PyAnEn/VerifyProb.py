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
# Class definition for probabilistic forecast verification
#

from random import sample
import numpy as np

from .Verify import Verify
from .utils_verify import binarize_obs
from .utils_verify import calculate_roc
from .utils_verify import rank_histogram
from .utils_verify import calculate_reliability
from .utils_verify import _binned_spread_skill_create_split


class VerifyProb(Verify):
    def __init__(self, f, o, avg_axis=None, n_sample_members=None, boot_samples=None,
                 working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        self.n_sample_members = n_sample_members
        
        super().__init__(avg_axis, boot_samples, working_directory, start_from_scratch)
    
    def _crps(self): raise NotImplementedError
    def _prob_to_determ(self): raise NotImplementedError
    def _prob_to_variance(self): raise NotImplementedError
    def _prob_to_ens(self): raise NotImplementedError
    def _cdf(self, over=None, below=None): raise NotImplementedError
        
    ###################
    # Private Methods #
    ###################
    
    ###### Metric Methods ######
    
    def _error(self):
        return self._prob_to_determ() - self.o
    
    def _sq_error(self):
        return (self._prob_to_determ() - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self._prob_to_determ() - self.o)
    
    def _rank_hist(self):
        return rank_histogram(f=self._prob_to_ens(), o=self.o, ensemble_axis=-1)
    
    def _spread(self):
        ens = self._prob_to_ens()
        return np.max(ens, axis=-1) - np.min(ens, axis=-1)
    
    def _brier(self, over, below):
        brier = self.cdf(over=over, below=below) - binarize_obs(self.o, over=over, below=below)
        return brier ** 2
    
    def _binned_spread_skill(self, nbins=15):
        
        # Calculate variances and squared errors
        ab_error = self._metric_workflow_1('ab_error', self._ab_error)
        variance = self._metric_workflow_1('variance', self._prob_to_variance)
        
        return _binned_spread_skill_create_split(variance, ab_error, nbins=nbins, sample_axis=self.avg_axis)
    
    def _reliability(self, nbins=15, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self._cdf(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_reliability(f_prob, o_binary, nbins)
    
    def _roc(self, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self._cdf(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_roc(f_prob, o_binary)
    
    def _sharpness(self, over=None, below=None):
        return self._cdf(over=over, below=below)
    
    ###### Other Methods ######
    
    def _validate(self):
        super()._validate()
    
        # Check forecasts and observations
        assert hasattr(self.f, 'keys'), 'f should be dict-like'
        assert isinstance(self.o, np.ndarray)
        
        # Check dimensions
        o_shape = list(self.o.shape)
        
        for k in self.f.keys():
            f_shape = list(self.f[k].shape)
            assert f_shape == o_shape, 'Shape mismatch: f[{}] ({}) and o ({})'.format(k, f_shape, o_shape)
        
        # Check number of ensemble members to sample
        if self.n_sample_members is not None:
            assert isinstance(self.n_sample_members, int)
