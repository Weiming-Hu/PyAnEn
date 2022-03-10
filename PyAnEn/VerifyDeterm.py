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
        
        self.f = f
        self.o = o
        
        super().__init__(avg_axis=avg_axis, boot_samples=boot_samples,
                         working_directory=working_directory,
                         start_from_scratch=start_from_scratch)
    
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
    
    def __str__(self):
        msg = super().__str__()
        msg += '\nForecasts (f): {}'.format(self.f.shape)
        msg += '\nObservations (o): {}'.format(self.o.shape)
        return msg
