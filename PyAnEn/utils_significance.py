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
# Date of Creation: 2022/04/18                              #
#############################################################
#
# This script defines functions for significance tests.
#

import numpy as np

from scipy.stats import t


def autocovariance(d, k):
    return np.mean((d[k:] - d.mean()) * (d[:(len(d) - k)] - d.mean()))


def diebold_mariano_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    """
    Code adopted from https://github.com/johntwk/Diebold-Mariano-Test/blob/master/dm_test.py
    
    Author   : John Tsang
    Date     : December 7th, 2017
    Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
            forecast accuracy
    Input    : 1) actual_lst: the list of actual values
            2) pred1_lst : the first list of predicted values
            3) pred2_lst : the second list of predicted values
            4) h         : the number of stpes ahead
            5) crit      : a string specifying the criterion 
                                i)  MSE : the mean squared error
                            ii)  MAD : the mean absolute deviation
                            iii) MAPE : the mean absolute percentage error
                            iv) poly : use power function to weigh the errors
            6) poly      : the power for crit power 
                            (it is only meaningful when crit is "poly")
    Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
            2) h must be an integer and it must be greater than 0 and less than 
                the length of actual_lst.
            3) crit must take the 4 values specified in Input
            4) Each value of actual_lst, pred1_lst and pred2_lst must
                be numerical values. Missing values will not be accepted.
            5) power must be a numerical value.
    Return   : a named-tuple of 2 elements
            1) p_value : the p-value of the DM test
            2) DM      : the test statistics of the DM test

    References:

    Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of 
    prediction mean squared errors. International Journal of forecasting, 
    13(2), 281-291.

    Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, 
    Journal of business & economic statistics 13(3), 253-264.
    """
    
    # Sanity check
    assert isinstance(actual_lst, np.ndarray) and isinstance(actual_lst, np.ndarray) and isinstance(actual_lst, np.ndarray), 'actual_lst, pred1_lst and pred2_lst must be numpy arrays!'
    actual_lst = actual_lst.ravel()
    pred1_lst = pred1_lst.ravel()
    pred2_lst = pred2_lst.ravel()
    
    lst_size = len(actual_lst)
    assert lst_size == len(pred1_lst) == len(pred2_lst), 'Lengths of actual_lst, pred1_lst and pred2_lst must match!'
    
    assert isinstance(h, int), 'The type of the number of steps ahead (h) must be an integer!'
    assert h >= 1, 'The number of steps ahead (h) should be no smaller than 1!'
    assert h < len(actual_lst), 'The number of steps ahead (h) is too large!'
    
    assert crit in ['MSE', 'MAPE', 'MAD', 'poly'], 'The criterion ({}) is not supported!'.format(crit)
    
    # construct loss-differential according to crit
    if crit == 'MSE':
        e1_lst = (actual_lst - pred1_lst) ** 2
        e2_lst = (actual_lst - pred2_lst) ** 2
        
    elif crit == 'MAD':
        e1_lst = np.abs(actual_lst - pred1_lst)
        e2_lst = np.abs(actual_lst - pred2_lst)
        
    elif (crit == "MAPE"):
        e1_lst = np.abs((actual_lst - pred1_lst) / actual_lst)
        e2_lst = np.abs((actual_lst - pred2_lst) / actual_lst)
        
    elif (crit == "poly"):
        e1_lst = (actual_lst - pred1_lst) ** power
        e2_lst = (actual_lst - pred2_lst) ** power
            
    d_lst = e1_lst - e2_lst
    
    # Find autocovariance
    gamma = np.array([autocovariance(d_lst, lag) for lag in range(0, h)])
    
    # Construct DM test statistics
    V_d = (gamma[0] + 2 * gamma[1:].sum()) / lst_size
    DM_stat = d_lst.mean() / np.sqrt(V_d)
    
    # Adjustment
    harvey_adj = np.sqrt((lst_size + 1 - 2 * h + h * (h-1) / lst_size) / lst_size)
    DM_stat = harvey_adj * DM_stat
    
    # Find p-value
    p_value = 2 * t.cdf(-np.abs(DM_stat), df = lst_size - 1)
    
    return DM_stat, p_value
