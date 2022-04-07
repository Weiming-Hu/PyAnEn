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
import shutil
import pickle

import numpy as np

from distutils import util
from .utils_verify import boot_arr
from .utils_verify import _reliability_agg_boot
from .utils_verify import _reliability_agg_no_boot
from .utils_verify import _binned_spread_skill_agg_boot
from .utils_verify import _binned_spread_skill_agg_no_boot


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
    def _variance(self): raise NotImplementedError
    def _crps(self): raise NotImplementedError
    def _rank_hist(self): raise NotImplementedError
    def _spread(self): raise NotImplementedError
    def _brier(self, over, below): raise NotImplementedError
    def _binned_spread_skill(self, nbins=15): raise NotImplementedError
    def _reliability(self, nbins=15, over=None, below=None): raise NotImplementedError
    def _roc(self, over=None, below=None): raise NotImplementedError
    def _sharpness(self, over=None, below=None): raise NotImplementedError
    def _iou_determ(self, over=None, below=None): raise NotImplementedError
    def _iou_prob(self, over=None, below=None): raise NotImplementedError
    def _cdf(self, over=None, below=None): raise NotImplementedError
    def _f_determ(self): raise NotImplementedError
        
    ##################
    # Metric Methods #
    ##################
    
    def rank_hist(self, save_name='rank'): return self._metric_workflow_1(save_name, self._rank_hist)
    def variance(self, save_name='variance'): return self._metric_workflow_1(save_name, self._variance)
    def f_determ(self, save_name='f_determ'): return self._metric_workflow_1(save_name, self._f_determ)
    def roc(self, over=None, below=None, save_name='roc'): return self._metric_workflow_1(self.to_name(save_name, over=over, below=below), self._roc, over=over, below=below)
    def sharpness(self, over=None, below=None, save_name='sharpness'): return self._metric_workflow_1(self.to_name(save_name, over=over, below=below), self._sharpness, over=over, below=below)
    def iou_determ(self, over=None, below=None, save_name='iou_determ'): return self._metric_workflow_1(self.to_name(save_name, over=over, below=below), self._iou_determ, over=over, below=below)
    def iou_prob(self, over=None, below=None, save_name='iou_prob'): return self._metric_workflow_1(self.to_name(save_name, over=over, below=below), self._iou_prob, over=over, below=below)
    def cdf(self, over=None, below=None, save_name='cdf'): return self._metric_workflow_1(self.to_name(save_name, over=over, below=below), self._cdf, over=over, below=below)
    
    def crps(self, save_name='crps'): return self._metric_workflow_2(save_name, self._crps)
    def error(self, save_name='error'): return self._metric_workflow_2(save_name, self._error)
    def spread(self, save_name='spread'): return self._metric_workflow_2(save_name, self._spread)
    def sq_error(self, save_name='sq_error'): return self._metric_workflow_2(save_name, self._sq_error)
    def ab_error(self, save_name='ab_error'): return self._metric_workflow_2(save_name, self._ab_error)
    def brier(self, over=None, below=None, save_name='brier'): return self._metric_workflow_2(self.to_name(save_name, over=over, below=below), self._brier, over=over, below=below)
    
    def reliability(self, nbins=15, over=None, below=None, save_name='rel'): return self._metric_workflow_3(self.to_name(save_name, nbins=nbins, over=over, below=below), self._reliability, self.post_reliability, nbins=nbins, over=over, below=below)
    def binned_spread_skill(self, nbins=15, save_name='ss'): return self._metric_workflow_3(self.to_name(save_name, nbins=nbins), self._binned_spread_skill, self.post_binned_spread_skill, nbins=nbins)

    def rmse(self, save_name='sq_error'):
        return np.sqrt(self.sq_error(save_name=save_name))
    
    ##################
    # Static Methods #
    ##################
    
    @staticmethod
    def reliability_index(freqs):
        freqs_sum = np.sum(freqs)
        
        if freqs_sum != 1:
            freqs = freqs / freqs_sum
            
        return np.sum(np.abs(freqs - 1 / np.arange(2, len(freqs) + 2)))
    
    ##########################
    # Postprocessing Methods #
    ##########################
    
    def boot_or_avg(self, metric):
        # Deals with averaging or bootstraping
        
        if self.boot_samples is None:
            # Only average over axis
            if util.strtobool(os.environ['pyanen_skip_nan']):
                return np.nanmean(metric, axis=self.avg_axis)
            else:
                return metric.mean(axis=self.avg_axis)
        
        else:
            # Use bootstraping to sample from averaging axis and create CI
            return boot_arr(metric, sample_axis=self.avg_axis, n_samples=self.boot_samples)
        
    def post_binned_spread_skill(self, metric):
        # Deals with averaging or bootstraping for spread skill arrays
        
        if self.boot_samples is None:
            # Only average over axis
            return _binned_spread_skill_agg_no_boot(*metric)
        
        else:
            # Use bootstraping to create CI
            return _binned_spread_skill_agg_boot(*metric, n_samples=self.boot_samples)
        
    def post_reliability(self, metric):
        # Deals with averaging or bootstraping for reliability diagram
        
        if self.boot_samples is None:
            # Only average over bins
            return _reliability_agg_no_boot(*metric)
        
        else:
            # Use bootstraping to create CI
            return _reliability_agg_boot(*metric, n_samples=self.boot_samples)
    
    ########################
    # Other Public Methods #
    ########################
    
    def to_name(self, save_name, special_keyword='LITERAL_', **kwargs):
        
        if isinstance(save_name, str): save_name = save_name.replace(' ', '_')
        else: return None
        
        if save_name[:len(special_keyword)] == special_keyword:
            return save_name[len(special_keyword):]
        
        strs = {}
        
        for k, v in kwargs.items():
            if hasattr(v, '__len__'):
                if isinstance(v, np.ndarray) and len(v.ravel()) > 5: return None
                elif len(v) > 5: return None
            else: strs[k] = str(v).replace(' ', '_')
        
        return '_'.join([save_name] + ['{}_{}'.format(k, v) for k, v in strs.items()])
    
    def set_avg_axis(self, x):
        self.avg_axis = x
        self._validate()
        return self
    
    def set_boot_samples(self, samples):
        self.boot_samples = samples
        self._validate()
        return self
        
    def disable_boot(self):
        self.boot_samples = None
        return self
        
    def enable_saving(self, working_directory, start_from_scratch=True):
        self.working_directory = working_directory
        self.start_from_scratch = start_from_scratch
        self._validate()
        return self
    
    def disable_saving(self):
        self.working_directory = None
        return self
            
    ###################
    # Private Methods #
    ###################
    
    def _validate_avg_axis(self):
        
        if self.avg_axis is not None:
            if isinstance(self.avg_axis, int):
                self.avg_axis = (self.avg_axis, )
            if isinstance(self.avg_axis, list):
                self.avg_axis = tuple(self.avg_axis)
            assert isinstance(self.avg_axis, tuple)
            
    def _validate_saving(self):
        
        if self.working_directory:
            assert isinstance(self.working_directory, str)
            self.working_directory = os.path.expanduser(self.working_directory)
            
            if os.path.exists(self.working_directory):
                if self.start_from_scratch:
                    shutil.rmtree(self.working_directory)
                    os.makedirs(self.working_directory)
            else:
                os.makedirs(self.working_directory)
    
    def _validate_boot_samples(self):
        if self.boot_samples is not None:
            assert isinstance(self.boot_samples, int)
    
    def _validate(self):
        
        # Check average axis
        self._validate_avg_axis()
            
        # Check boot samples
        self._validate_boot_samples()
            
        # Check start_from_scratch
        assert isinstance(self.start_from_scratch, bool)
            
        # Check working directory
        self._validate_saving()
    
    def _save(self, name, arr):
        if self.working_directory and isinstance(name, str):
            path = os.path.join(self.working_directory, name + '.pkl')
            with open(path, 'wb') as f:
                pickle.dump(arr, f)
        
    def _load(self, name):
        if self.working_directory and isinstance(name, str):
            path = os.path.join(self.working_directory, name + '.pkl')
            
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            
        return None
    
    def _metric_workflow_1(self, save_name, func, **kwargs):
        metric = self._load(save_name)
        
        if metric is None:
            metric = func(**kwargs)
            self._save(save_name, metric)
            
        return metric
    
    def _metric_workflow_2(self, save_name, func, **kwargs):
        metric = self._metric_workflow_1(save_name, func, **kwargs)
        return self.boot_or_avg(metric)
    
    def _metric_workflow_3(self, save_name, func, post_func, **kwargs):
        metric = self._metric_workflow_1(save_name, func, **kwargs)
        return post_func(metric)
    
    def __repr__(self):
        msg = '=============== PyAnEn::{} ==============='.format(type(self).__name__)
        
        if hasattr(self, 'f'):
            msg += '\nForecast (f): {}'.format(Verify._format(self.f))
            
        if hasattr(self, 'o'):
            msg += '\nObservations (o): {}'.format(Verify._format(self.o))
        
        msg += '\nAverage/Sample axis (avg_axis): {}'.format(self.avg_axis)
        msg += '\nBootstrap samples (boot_samples): {}'.format(self.boot_samples)
        msg += '\nSave intermediate data at (working_directory): {}'.format(self.working_directory)
        msg += '\nIgnore saved intermediate data (start_from_scratch): {}'.format(self.start_from_scratch)
        
        return msg
    
    @staticmethod
    def _format(obj, indent=''):
        if not hasattr(obj, "__len__"):
            return '{}{}'.format(indent, obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return '{}sequence [{}]'.format(indent, len(obj))
        elif isinstance(obj, np.ndarray):
            return '{}array {}'.format(indent, obj.shape)
        elif isinstance(obj, dict):
            return '\n{}  - '.format(indent).join([' dict'] + ['{}:{}'.format(k, Verify._format(v, ' ')) for k, v in obj.items()])
        else:
            return '** UNKNOWN TYPE **'
        
