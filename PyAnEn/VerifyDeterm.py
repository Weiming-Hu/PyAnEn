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
from .utils_special_boot import corr
from .utils_verify import iou_determ


class VerifyDeterm(Verify):
    def __init__(self, f, o, avg_axis=None, boot_samples=None,
                 working_directory=None, start_from_scratch=True):
        
        self.f = f
        self.o = o
        
        super().__init__(avg_axis=avg_axis, boot_samples=boot_samples,
                         working_directory=working_directory,
                         start_from_scratch=start_from_scratch)
    
    ##################
    # Metric Methods #
    ##################
    
    def _error(self):
        return self.f - self.o
    
    def _sq_error(self):
        return (self.f - self.o) ** 2
    
    def _ab_error(self):
        return np.abs(self.f - self.o)
    
    def _iou_determ(self, over=None, below=None):
        return iou_determ(self.f, self.o, axis=self.avg_axis, over=over, below=below)
    
    def _f_determ(self):
        return np.copy(self.f)
    
    def _corr(self):
        return corr(self.f_determ(), self.o, self.avg_axis, self.boot_samples)
    
    ###################
    # Private Methods #
    ###################
    
    def _validate(self):
        super()._validate()
        
        # Check forecasts and observations
        assert isinstance(self.f, np.ndarray)
        assert isinstance(self.o, np.ndarray)
        
        # Check dimensions
        f_shape, o_shape = list(self.f.shape), list(self.o.shape)
        assert f_shape == o_shape, 'Shape mismatch: f ({}) and o ({})'.format(f_shape, o_shape)
    