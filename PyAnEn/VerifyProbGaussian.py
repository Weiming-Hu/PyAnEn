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

from .VerifyProb import VerifyProb
from .utils_dist import sample_dist_gaussian, cdf_gaussian


class VerifyProbGaussian(VerifyProb):
    def __init__(self, f, o, move_sampled_ens_axis=-1, truncated=False, avg_axis=None, n_sample_members=None,
                 boot_samples=None, working_directory=None, start_from_scratch=True):
        
        self.truncated = truncated
        
        super().__init__(f, o, move_sampled_ens_axis, avg_axis, n_sample_members, boot_samples, working_directory, start_from_scratch)
        
    def set_truncation(self, use_truncation):
        self.truncated = use_truncation
        self._validate_truncation()
        return self
    
    def _validate_truncation(self):
        assert isinstance(self.truncated, bool)
        
    def _validate(self):
        super()._validate()
        self._validate_truncation()
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        
    def _prob_to_determ(self):
        return self.f['mu']
    
    def _prob_to_variance(self):
        return self.f['sigma'] ** 2
    
    def _prob_to_ens(self):
        assert self.n_sample_members is not None, 'Set the number of members to sample, e.g., obj.n_sample_members = 15'
        return sample_dist_gaussian(self.f['mu'], self.f['sigma'], self.n_sample_members, self.move_sampled_ens_axis)
    
    def _cdf(self, over=None, below=None):
        return cdf_gaussian(self.f['mu'], self.f['sigma'], over, below, self.truncated)
    
    def _crps(self):
        if self.truncated:
            # TODO: Implement the CRPS with truncated gaussian
            raise NotImplementedError
        else:
            return ps.crps_gaussian(self.o, mu=self.f['mu'], sig=self.f['sigma'])
    
    def __str__(self):
        msg = super().__str__()
        msg += '\nTruncated distribution (truncated): {}'.format(', '.join(self.truncated))
        return msg
