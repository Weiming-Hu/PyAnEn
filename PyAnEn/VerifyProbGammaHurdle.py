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

from .VerifyProb import VerifyProb
from .utils_dist import cdf_gamma_hurdle
from .utils_approximation import Integration


class VerifyProbGammaHurdle(VerifyProb):
    
    def __init__(self, f, o, move_sampled_ens_axis=-1, truncated=False,
                 pit_randomize_zero_ranks=True, avg_axis=None,
                 n_sample_members=None, clip_member_to_zero=None, n_approx_bins=20, 
                 integration_range=None, boot_samples=None,
                 working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        self.n_approx_bins = n_approx_bins
        self.integration_range = integration_range
        
        super().__init__(move_sampled_ens_axis, truncated, pit_randomize_zero_ranks, 
                         avg_axis, n_sample_members, clip_member_to_zero, boot_samples, working_directory, start_from_scratch)
    
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
    
        # Check forecasts and observations
        assert hasattr(self.f, 'keys'), 'f should be dict-like'
        assert isinstance(self.o, np.ndarray)
        
        assert 'pop' in self.f.keys()
        assert 'mu' in self.f.keys()
        assert 'sigma' in self.f.keys()
        assert isinstance(self.n_approx_bins, int)
        
        if self.integration_range is not None:
            assert len(self.integration_range) == 2
    
    def _validate_truncation(self):
        super()._validate_truncation()
        assert not self.truncated, 'Truncation is currently not implemented.'
        
    def _variance(self):
        return Integration(verifier=self, integration_range=self.integration_range, nbins=self.n_approx_bins).variance()
    
    def _cdf(self, over=None, below=None):
        return cdf_gamma_hurdle(self.f['pop'], self.f['mu'], self.f['sigma'], over=over, below=below)
    
    def _f_determ(self):
        return Integration(verifier=self, integration_range=self.integration_range, nbins=self.n_approx_bins).mean()
    
    def _crps(self):
        return Integration(verifier=self, integration_range=self.integration_range, nbins=self.n_approx_bins).crps()
    
