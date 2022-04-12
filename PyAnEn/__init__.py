import os

from .version import __version__
from .VerifyDeterm import VerifyDeterm
from .VerifyEnsemble import VerifyEnsemble
from .VerifyProbGamma import VerifyProbGamma
from .VerifyProbGaussian import VerifyProbGaussian
from .VerifyProbGammaHurdle import VerifyProbGammaHurdle

__title__ = "PyAnEn"
__author__ = "Weiming Hu"

Defaults = {
    'pyanen_ens_to_prob_method': 'moments',
    'pyanen_kde_bandwidth': 0.01,
    'pyanen_kde_kernel': 'gaussian',
    'pyanen_kde_samples': 1000,
    'pyanen_kde_multiply_spread': 5,
    'pyanen_boot_confidence': 0.95,
    'pyanen_boot_repeats': 1000,
    'pyanen_boot_samples': 300,
    'pyanen_tqdm_disable': True,
    'pyanen_tqdm_leave': False,
    'pyanen_tqdm_workers': 1,
    'pyanen_tqdm_map_axis': -1,
    'pyanen_tqdm_chunksize': 1,
    'pyanen_use_tensorflow_math': False,
    'pyanen_skip_nan': False,
    'pyanen_split_crps_ensemble_along_0': False,
    'pyanen_integrate_range_multiplier': 5,
    'pyanen_integrate_with_less_memory': False,
    'pyanen_numerical_integration_tmp_dir': '',
}

for k, v in Defaults.items():
    if k not in os.environ:
        os.environ[k] = str(v)


def print_settings():
    for k in Defaults:
        print("os.environ['{}'] = '{}'".format(k, os.environ[k]))
        
