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
# Date of Creation: 2022/03/14                              #
#############################################################
#
# This script defines the class for a truncated Gamma distribution
#

import numpy as np
import scipy.special as sc

from scipy import stats
from scipy.stats import rv_continuous
from scipy.special import gammainc, gammaincinv


class truncgamma_gen(rv_continuous):
    # References:
    # https://github.com/duncandc/stat_utils/blob/master/trunc_gamma.py
    # https://en.wikipedia.org/wiki/Truncated_distribution
    # https://stats.stackexchange.com/a/534340

    def _argcheck(self, a, b, s):
        self.a = a  # lower bound
        self.b = b  # upper bound
        self.s = s
        return (self.a >= 0.0) & (self.b > self.a)
    
    def _logpdf(self, x, a):
        return sc.xlogy(a-1.0, x) - x - sc.gammaln(a)

    def _pdf(self, x, a, b, s):
        result = np.exp(self._logpdf(x, s)) / (sc.gammainc(s, b) - sc.gammainc(s, a))
        
        mask = (x<a) | (x>b)
        result[mask] = 0.0
        return result

    def _cdf(self, x, a, b, s):
        
        P_x = sc.gammainc(s, x)
        P_a = sc.gammainc(s, a)
        P_b = sc.gammainc(s, b)
        result = (P_x - P_a) / (P_b - P_a)
        
        mask = (x<a) | (x>b)
        result[mask] = 0.0
        return result
    
    def _ppf(self, q, a, b, s):
        P_a = gammainc(s, a)
        P_b = gammainc(s, b)
        return gammaincinv(s, q * (P_b - P_a) + P_a)
    
    def _rvs(self, a, b, s, size=None, random_state=None):
        q = stats.uniform().rvs(size=size)
        return self._ppf(q, a, b, s)
