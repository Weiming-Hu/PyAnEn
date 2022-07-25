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

import os
import numpy as np

from scipy import stats
from distutils import util
from .Verify import Verify
from .utils_special_boot import brier_decomp
from .utils_verify import iou_prob
from .utils_verify import iou_determ
from .utils_verify import binarize_obs
from .utils_verify import calculate_roc
from .utils_verify import rank_histogram
from .utils_verify import _reliability_split
from .utils_verify import _binned_spread_skill_create_split


class VerifyProb(Verify):
    
    def __init__(self, move_sampled_ens_axis=-1, truncated=False, pit_randomize_zero_ranks=True,
                 avg_axis=None, n_sample_members=None, clip_member_to_zero=None,
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.truncated = truncated
        self.n_sample_members = n_sample_members
        self.clip_member_to_zero = clip_member_to_zero
        self.move_sampled_ens_axis = move_sampled_ens_axis
        self.pit_randomize_zero_ranks = pit_randomize_zero_ranks
        
        super().__init__(avg_axis, boot_samples, working_directory, start_from_scratch)
    
    def prob_to_ens(self):
        return self._metric_workflow_1('prob_to_ens', self._prob_to_ens)
    
    def pit(self):
        return self._metric_workflow_1('pit', self._pit)
    
    def _crps(self): raise NotImplementedError
    def _variance(self): raise NotImplementedError
    def _f_determ(self): raise NotImplementedError
    def _prob_to_ens(self): raise NotImplementedError
    def _cdf(self, over=None, below=None): raise NotImplementedError
    def _corr(self): raise NotImplementedError
    
    ##################
    # Public Methods #
    ##################
    
    def set_move_sampled_ens_axis(self, axis):
        self.move_sampled_ens_axis = axis
        self._validate()
        return self
    
    def set_ensemble_members(self, n):
        self.n_sample_members = n
        self._validate()
        return self
    
    def set_member_clip(self, v):
        self.clip_member_to_zero = v
        self._validate()
        return self
        
    def set_truncation(self, use_truncation):
        self.truncated = use_truncation
        self._validate()
        return self
    
    ###################
    # Private Methods #
    ###################
    
    ###### Metric Methods ######
    
    def _error(self):
        return self.f_determ() - self.o
    
    def _sq_error(self):
        return (self.f_determ() - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self.f_determ() - self.o)
    
    def _rank_hist(self):
        if self.n_sample_members is None:
            # Use probability integral transform
            return self.pit()
        
        else:
            # Use random sampling
            return rank_histogram(f=self.prob_to_ens(), o=self.o, ensemble_axis=-1)
    
    def _spread(self):
        ens = self.prob_to_ens()
        if util.strtobool(os.environ['pyanen_skip_nan']):
            return np.nanmax(ens, axis=-1) - np.nanmin(ens, axis=-1)
        else:
            return np.max(ens, axis=-1) - np.min(ens, axis=-1)
    
    def _brier(self, over, below):
        brier = self.cdf(over=over, below=below) - binarize_obs(self.o, over=over, below=below)
        return brier ** 2
    
    def _brier_decomp(self, over, below):
        f = self.cdf(over=over, below=below)
        o = binarize_obs(self.o, over=over, below=below)
        return brier_decomp(f, o, self.avg_axis, self.boot_samples)
    
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
        
        return _reliability_split(f_prob, o_binary, nbins)
    
    def _roc(self, over=None, below=None):
        
        # Calculate forecasted probability and binarized observations
        f_prob = self.cdf(over=over, below=below)
        o_binary = binarize_obs(self.o, over=over, below=below)
        
        return calculate_roc(f_prob, o_binary)
    
    def _sharpness(self, over=None, below=None):
        return self.cdf(over=over, below=below)
    
    def _iou_determ(self, over=None, below=None):
        return iou_determ(self.f_determ(), self.o, axis=self.avg_axis, over=over, below=below)
    
    def _iou_prob(self, over=(None, None), below=(None, None)):
        
        assert isinstance(over, tuple) or isinstance(over, list)
        assert isinstance(below, tuple) or isinstance(below, list)
        assert len(over) == 2 or len(below) == 2
        
        f_prob = self.cdf(over=over[0], below=below[0])
        o_binary = binarize_obs(self.o, over=over[0], below=below[0])
        
        return iou_prob(f_prob, o_binary, axis=self.avg_axis, over=over[1], below=below[1])
    
    def _pit(self):
        # Call the private functions of CDF because results should not be automatically saved
        # when an array, rather than a scalar, is used as the threshold (below / over). 
        #
        ranks = self._cdf(over=None, below=self.o)
        
        if self.pit_randomize_zero_ranks:
            mask = self.o == 0
            ranks[mask] = stats.uniform(loc=0, scale=ranks[mask]).rvs()
        
        return ranks
        
    ###### Other Methods ######
    
    def _validate_sample_members(self):
        if self.n_sample_members is not None:
            assert isinstance(self.n_sample_members, int)
    
    def _validate_truncation(self):
        assert isinstance(self.truncated, bool)
            
    def _validate_clip_member_to_zero(self):
        if self.clip_member_to_zero is not None:
            assert isinstance(self.clip_member_to_zero, float)
    
    def _validate(self):
        super()._validate()
        
        assert isinstance(self.move_sampled_ens_axis, int)
        assert isinstance(self.pit_randomize_zero_ranks, bool)
        
        # Check number of ensemble members to sample
        self._validate_sample_members()
        self._validate_clip_member_to_zero()
        self._validate_truncation()
    
    def __repr__(self):
        msg = super().__repr__()
        msg += '\nEnsemble members to sample (n_sample_members): {}'.format(self.n_sample_members)
        msg += '\nMove generated ensemble axis to (move_sampled_ens_axis): {}'.format(self.move_sampled_ens_axis)
        msg += '\nTruncated (truncated): {}'.format(self.truncated)
        msg += '\nRandomize zero ranks (pit_randomize_zero_ranks): {}'.format(self.pit_randomize_zero_ranks)
        return msg
