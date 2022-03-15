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
# Class definition for probabilistic forecast verification with a Censored, Shifted Gamma Distribution
#

from .VerifyProb import VerifyProb
from .utils_verify import crps_csgd
from .utils_dist import sample_dist_csgd, cdf_csgd


class VerifyProbCSGD(VerifyProb):
    
    def __init__(self, f, o, move_sampled_ens_axis=-1, avg_axis=None,
                 n_sample_members=None, clip_member_to_zero=None,
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        super().__init__(f, o, move_sampled_ens_axis, avg_axis, n_sample_members, clip_member_to_zero, boot_samples, working_directory, start_from_scratch)
        
    def _validate(self):
        super()._validate()
        
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        assert 'shift' in self.f.keys()
        assert 'unshifted_mu' in self.f.keys()
        
    def _prob_to_determ(self):
        return self.f['mu']
    
    def _prob_to_variance(self):
        return self.f['sigma'] ** 2
    
    def _prob_to_ens(self):
        assert self.n_sample_members is not None, 'Set the number of members to sample, e.g., obj.set_ensemble_members(15)'
        ens = sample_dist_csgd(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], self.n_sample_members, self.move_sampled_ens_axis)
        if self.clip_member_to_zero is not None:
            ens[ens < self.clip_member_to_zero] = 0
        return ens
    
    def _cdf(self, over=None, below=None):
        return cdf_csgd(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], over, below)
    
    def _crps(self):
        return crps_csgd(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], self.o, reduce_sum=False)
    