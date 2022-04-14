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
# Class definition for probabilistic forecast verification with a Gamma Distribution
#

import numpy as np

from .VerifyProb import VerifyProb
from .utils_crps import crps_gamma
from .utils_special_boot import corr
from .utils_dist import sample_dist_gamma, cdf_gamma


class VerifyProbGamma(VerifyProb):
    
    def __init__(self, f, o, move_sampled_ens_axis=-1, truncated=False,
                 pit_randomize_zero_ranks=True, avg_axis=None,
                 n_sample_members=None, clip_member_to_zero=None,
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        
        super().__init__(move_sampled_ens_axis, truncated, pit_randomize_zero_ranks, 
                         avg_axis, n_sample_members, clip_member_to_zero, boot_samples, working_directory, start_from_scratch)
        
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
        
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        assert 'shift' in self.f.keys()
        assert 'unshifted_mu' in self.f.keys()
        
    def _f_determ(self):
        return np.copy(self.f['mu'])
    
    def _variance(self):
        return self.f['sigma'] ** 2
    
    def _prob_to_ens(self):
        assert self.n_sample_members is not None, 'Set the number of members to sample, e.g., obj.set_ensemble_members(15)'
        
        ens = sample_dist_gamma(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], self.n_sample_members, self.move_sampled_ens_axis, self.truncated)
        
        if self.clip_member_to_zero is not None:
            ens[ens < self.clip_member_to_zero] = 0
            
        return ens
    
    def _cdf(self, over=None, below=None):
        assert (over is None) ^ (below is None), 'Must specify over or below'
        if below is None: return 1 - self.cdf(below=over)
        else: return cdf_gamma(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], below, truncated=self.truncated)
    
    def _crps(self):
        
        if self.truncated:
            raise NotImplementedError
        else:
            return crps_gamma(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], self.o, reduce_sum=False)
    
    def _corr(self):
        return corr(self.f_determ(), self.o, self.avg_axis, self.boot_samples)
    