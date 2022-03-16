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
# Class definition for probabilistic forecast verification with a Gaussian Distribution
#

import properscoring as ps

from scipy import stats
from .VerifyProb import VerifyProb
from .utils_crps import crps_truncated_gaussian
from .utils_dist import sample_dist_gaussian, cdf_gaussian


class VerifyProbGaussian(VerifyProb):
    def __init__(self, f, o, move_sampled_ens_axis=-1, truncated=False, avg_axis=None,
                 n_sample_members=None, clip_member_to_zero=None, 
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        super().__init__(f, o, move_sampled_ens_axis, truncated, avg_axis, n_sample_members, clip_member_to_zero, boot_samples, working_directory, start_from_scratch)
        
    def _validate(self):
        super()._validate()
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        
    def _prob_to_determ(self):
        return self.f['mu']
    
    def _prob_to_variance(self):
        return self.f['sigma'] ** 2
    
    def _prob_to_ens(self):
        assert self.n_sample_members is not None, 'Set the number of members to sample, e.g., obj.n_sample_members = 15'
        
        ens = sample_dist_gaussian(self.f['mu'], self.f['sigma'], self.n_sample_members, self.move_sampled_ens_axis, self.truncated)
        
        if self.clip_member_to_zero is not None:
            ens[ens < self.clip_member_to_zero] = 0
            
        return ens
    
    def _cdf(self, over=None, below=None):
        return cdf_gaussian(self.f['mu'], self.f['sigma'], over, below, self.truncated)
    
    def _crps(self):
        
        if self.truncated:
            return crps_truncated_gaussian(self.o, mu=self.f['mu'], scale=self.f['sigma'], l=0)
        else:
            return ps.crps_gaussian(self.o, mu=self.f['mu'], sig=self.f['sigma'])
    
    def _pit(self):
        ranks = cdf_gaussian(self.f['mu'], self.f['sigma'], over=None, below=self.o, truncated=self.truncated)
        
        mask = self.o == 0
        ranks[mask] = stats.uniform(loc=0, scale=ranks[mask]).rvs()
        
        return ranks
