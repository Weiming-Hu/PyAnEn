# "`-''-/").___..--''"`-._
#  (`6_ 6  )   `-.  (     ).`-.__.`)   WE ARE ...
#  (_Y_.)'  ._   )  `._ `. ``-..-'    PENN STATE!
#    _ ..`--'_..-_/  /--'_.' ,'
#  (il),-''  (li),'  ((!.-'
#
# Author: Weiming Hu <weiming@psu.edu>
#
#         Geoinformatics and Earth Observation Laboratory (http://geolab.psu.edu)
#         Department of Geography and Institute for Computational and Data Sciences
#         The Pennsylvania State University
#

import os

import numpy as np

from .utils_verify import boot_arr, _binned_spread_skill_agg_boot, _binned_spread_skill_agg_no_boot


# For deterministic forecasts: f and o are both numpy arrays; f.shape == o.shape
# For ensemble forecasts: f and o are both numpy arrays; f.shape - ensemble axis == o.shape
# For probabilistic forecasts: o is an numpy array; f is a dictionary of numpy arrays with shapes of 0

class Verify:
    def __init__(self, avg_axis=None, boot_samples=None, working_directory=None, start_from_scratch=True):
        
        # Initialization
        self.avg_axis = avg_axis
        self.boot_samples = boot_samples
        self.working_directory = working_directory
        self.start_from_scratch = start_from_scratch
        
        self._validate()
    
    def _error(self): raise NotImplementedError
    def _sq_error(self): raise NotImplementedError
    def _ab_error(self): raise NotImplementedError
    def _crps(self): raise NotImplementedError
    def _rank_hist(self): raise NotImplementedError
    def _spread(self): raise NotImplementedError
    def _brier(self, over, below): raise NotImplementedError
    def _binned_spread_skill(self, nbins=15): raise NotImplementedError
    def _reliability(self, nbins=15, over=None, below=None): raise NotImplementedError
    def _roc(self, over=None, below=None): raise NotImplementedError
    def _sharpness(self, over=None, below=None): raise NotImplementedError
        
    ##################
    # Metric Methods #
    ##################
    
    # TODO: how to make it easier to visulize rank histogram results?
    def rank_hist(self, save_name='rank'): return self._metric_workflow_1(save_name, self._rank_hist)
    def reliability(self, nbins=15, over=None, below=None, save_name='rel'): return self._metric_workflow_1('_'.join([save_name, str(nbins), str(over), str(below)]), self._reliability, nbins=nbins, over=over, below=below)
    def roc(self, over=None, below=None, save_name='roc'): return self._metric_workflow_1('_'.join([save_name, str(over), str(below)]), self._roc, over=over, below=below)
    def sharpness(self, over=None, below=None, save_name='sharpness'): return self._metric_workflow_1('_'.join([save_name, str(over), str(below)]), self._sharpness, over=over, below=below)
    def crps(self, save_name='crps'): return self._metric_workflow_2(save_name, self._crps)
    def error(self, save_name='error'): return self._metric_workflow_2(save_name, self._error)
    def spread(self, save_name='spread'): return self._metric_workflow_2(save_name, self._spread)
    def sq_error(self, save_name='sq_error'): return self._metric_workflow_2(save_name, self._sq_error)
    def ab_error(self, save_name='ab_error'): return self._metric_workflow_2(save_name, self._ab_error)
    def brier(self, over=None, below=None, save_name='brier'): return self._metric_workflow_2('_'.join([save_name, str(over), str(below)]), self._brier, over=over, below=below)
    def binned_spread_skill(self, nbins=15, save_name='ss'): return self._metric_workflow_3('_'.join([save_name, str(nbins)]), self._binned_spread_skill, self.post_binned_spread_skill, nbins=nbins)

    def rmse(self, save_name='sq_error'):
        return np.sqrt(self.sq_error(save_name=save_name))
    
    ##################
    # Static Methods #
    ##################
    
    @staticmethod
    def to_skill_score(f_score, benchmark_score):
        assert f_score.shape == benchmark_score.shape
        return 1 - f_score / benchmark_score
    
    ##########################
    # Postprocessing Methods #
    ##########################
    
    def boot_or_avg(self, metric):
        # Deals with averaging or bootstraping
        
        if self.boot_samples is None:
            # Only average over axis
            return metric.mean(axis=self.avg_axis)
        
        else:
            # Use bootstraping to sample from averaging axis and create CI
            return boot_arr(metric, sample_axis=self.avg_axis)
        
    def post_binned_spread_skill(self, metric):
        # Deals with averaging or bootstraping for spread skill arrays
        
        if self.boot_samples is None:
            # Only average over axis
            return _binned_spread_skill_agg_no_boot(*metric)
        
        else:
            # Use bootstraping to create CI
            return _binned_spread_skill_agg_boot(*metric)
            
    
    ###################
    # Private Methods #
    ###################
    
    def _validate(self):
        
        # Check average axis
        if self.avg_axis is not None:
            if isinstance(self.avg_axis, int):
                self.avg_axis = (self.avg_axis, )
            assert isinstance(self.avg_axis, tuple)
            
        # Check boot samples
        if self.boot_samples is not None:
            assert isinstance(self.boot_samples, int)
            
        # Check start_from_scratch
        assert isinstance(self.start_from_scratch, bool)
            
        # Check working directory
        if self.working_directory:
            assert isinstance(self.working_directory, str)
            self.working_directory = os.path.expanduser(self.working_directory)
            
            if os.path.exists(self.working_directory):
                if self.start_from_scratch:
                    os.rmdir(self.working_directory)
                    os.mkdir(os.working_directory)
            else:
                os.mkdir(os.working_directory)
    
    def _save_npy(self, name, arr):
        if self.working_directory:
            path = os.path.join(self.working_directory, name, '.npy')
            np.save(path, arr)
        
    def _load_npy(self, name):
        if self.working_directory:
            path = os.path.join(self.working_directory, name, '.npy')
            
            if os.path.exists(path):
                return np.load(path)
            
        return None
    
    def _metric_workflow_1(self, save_name, func, **kwargs):
        metric = self._load_npy(save_name)
        
        if metric is None:
            metric = func(**kwargs)
            self._save_npy(save_name, metric)
            
        return metric
    
    def _metric_workflow_2(self, save_name, func, **kwargs):
        metric = self._metric_workflow_1(save_name, func, **kwargs)
        return self.boot_or_avg(metric)
    
    def _metric_workflow_3(self, save_name, func, post_func, **kwargs):
        metric = self._metric_workflow_1(save_name, func, **kwargs)
        return post_func(metric)
