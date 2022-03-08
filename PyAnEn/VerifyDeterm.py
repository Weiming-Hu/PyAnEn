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
# Class definition for deterministic forecast verification
#

import numpy as np

from .Verify import Verify


class VerifyDeterm(Verify):
    def __init__(self, f, o, avg_axis=None, boot_samples=None,
                 working_directory=None, start_from_scratch=True):
        super().__init__(avg_axis=avg_axis, boot_samples=boot_samples,
                         working_directory=working_directory,
                         start_from_scratch=start_from_scratch)
        
        self.f = f
        self.o = o
        
        self._validate()
    
    def _validate(self):
        super()._validate()
        
        # Check forecasts and observations
        assert isinstance(self.f, np.ndarray)
        assert isinstance(self.o, np.ndarray)
        
        # Check dimensions
        f_shape, o_shape = list(self.f.shape), list(self.o.shape)
        assert f_shape == o_shape, 'Shape mismatch: f ({}) and o ({})'.format(f_shape, o_shape)
    
    ##################
    # Metric Methods #
    ##################
    
    def _error(self):
        return self.f - self.o
    
    def _sq_error(self):
        return (self.f - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self.f - self.o)
