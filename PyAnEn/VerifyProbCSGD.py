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
from .utils_dist import sample_dist_csgd, cdf_csgd


class VerifyProbCSGD(VerifyProb):
    def __init__(self, f, o, shift_sign=1, move_sampled_ens_axis=-1, avg_axis=None, n_sample_members=None,
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.shift_sign = shift_sign
        self.move_sampled_ens_axis = move_sampled_ens_axis
        
        super().__init__(f, o, avg_axis, n_sample_members, boot_samples, working_directory, start_from_scratch)
        
    def _validate(self):
        super()._validate()
        
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        assert 'shift' in self.f.keys()
        assert 'unshifted_mu' in self.f.keys()
        
        assert self.shift_sign == 1 or self.shift_sign == -1, 'shift_sign must be 1 or -1. Got {}'.format(self.shift_sign)
        assert isinstance(self.move_sampled_ens_axis, int)
        
    def _prob_to_determ(self):
        return self.f['mu']
    
    def _prob_to_variance(self):
        return self.f['sigma'] ** 2
    
    def _prob_to_ens(self):
        assert self.n_sample_members is not None, 'Set the number of members to sample, e.g., obj.n_sample_members = 15'
        return sample_dist_csgd(
            self.f['unshifted_mu'], self.f['sigma'], self.f['shift'],
            self.shift_sign, self.n_sample_members, self.move_sampled_ens_axis)
    
    def _cdf(self, over=None, below=None):
        return cdf_csgd(self.f['unshifted_mu'], self.f['sigma'], self.f['shift'], self.shift_sign, over, below)
