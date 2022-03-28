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
    
    def _crps(self): raise NotImplementedError
    def _prob_to_determ(self): raise NotImplementedError
    def _prob_to_variance(self): raise NotImplementedError
    def _prob_to_ens(self): raise NotImplementedError
    def _cdf(self, over=None, below=None): raise NotImplementedError
    
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
    
    def _f_determ(self):
        return self._prob_to_determ()
    
    def _error(self):
        return self._prob_to_determ() - self.o
    
    def _sq_error(self):
        return (self._prob_to_determ() - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self._prob_to_determ() - self.o)
    
    def _rank_hist(self):
        if self.n_sample_members is None:
            # Use probability integral transform
            return self._pit()
        
        else:
            # Use random sampling
            return rank_histogram(f=self._prob_to_ens(), o=self.o, ensemble_axis=-1)
    
    def _spread(self):
        ens = self._prob_to_ens()
        if util.strtobool(os.environ['pyanen_skip_nan']):
            return np.nanmax(ens, axis=-1) - np.nanmin(ens, axis=-1)
        else:
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
        return iou_determ(self._prob_to_determ(), self.o, axis=self.avg_axis, over=over, below=below)
    
    def _iou_prob(self, over=(None, None), below=(None, None)):
        
        assert isinstance(over, tuple) or isinstance(over, list)
        assert isinstance(below, tuple) or isinstance(below, list)
        assert len(over) == 2 or len(below) == 2
        
        f_prob = self.cdf(over=over[0], below=below[0])
        o_binary = binarize_obs(self.o, over=over[0], below=below[0])
        
        return iou_prob(f_prob, o_binary, axis=self.avg_axis, over=over[1], below=below[1])
    
    def _pit(self):
        ranks = self.cdf(over=None, below=self.o)
        
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
    
    def __str__(self):
        msg = super().__str__()
        msg += '\nEnsemble members to sample (n_sample_members): {}'.format(self.n_sample_members)
        msg += '\nMove generated ensemble axis to (move_sampled_ens_axis): {}'.format(self.move_sampled_ens_axis)
        msg += '\nTruncated (truncated): {}'.format(self.truncated)
        msg += '\nRandomize zero ranks (pit_randomize_zero_ranks): {}'.format(self.pit_randomize_zero_ranks)
        return msg
