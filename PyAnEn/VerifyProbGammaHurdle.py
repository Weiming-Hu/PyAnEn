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
# Date of Creation: 2022/03/22                              #
#############################################################
#
# Class definition for probabilistic forecast verification with a Gamma Hurdle Model
#

import numpy as np

from scipy import stats
from .VerifyProb import VerifyProb
from .utils_dist import cdf_gamma_hurdle
from .utils_approximation import integrate


class VerifyProbGammaHurdle(VerifyProb):
    
    def __init__(self, f, o, move_sampled_ens_axis=-1, truncated=False, avg_axis=None,
                 n_sample_members=None, clip_member_to_zero=None, n_approx_bins=20, 
                 integration_range=None, boot_samples=None,
                 working_directory=None, start_from_scratch=True):
        
        self.n_approx_bins = n_approx_bins
        self.integration_range = integration_range
        
        super().__init__(f, o, move_sampled_ens_axis, truncated, avg_axis, n_sample_members, clip_member_to_zero, boot_samples, working_directory, start_from_scratch)
    
    def set_bins(self, nbins):
        self.n_approx_bins = nbins
        self._validate()
        return self
    
    def set_integration_range(self, vmin, vmax):
        self.integration_range = (vmin, vmax)
        self._validate()
        return self
    
    def _validate(self):
        super()._validate()
        
        assert 'pop' in self.f.keys()
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        assert isinstance(self.n_approx_bins, int)
        
        if self.integration_range is not None:
            assert len(self.integration_range) == 2
    
    def _validate_truncation(self):
        super()._validate_truncation()
        assert not self.truncated, 'Truncation is currently not implemented.'
        
    def _prob_to_variance(self):
        return np.copy(self.f['sigma']) ** 2
    
    def _cdf(self, over=None, below=None):
        return cdf_gamma_hurdle(self.f['pop'], self.f['mu'], self.f['sigma'], over=over, below=below)
    
    def _pit(self):
        ranks = cdf_gamma_hurdle(self.f['pop'], self.f['mu'], self.f['sigma'], over=None, below=self.o)
        
        mask = self.o == 0
        ranks[mask] = stats.uniform(loc=0, scale=ranks[mask]).rvs()
        
        return ranks

    def _prob_to_determ(self):
        return integrate(verifier=self, type='cdf', nbins=self.n_approx_bins, return_slices=False, integration_range=None)
    
    def _crps(self):
        return integrate(verifier=self, type='brier', nbins=self.n_approx_bins, return_slices=False, integration_range=None)
    