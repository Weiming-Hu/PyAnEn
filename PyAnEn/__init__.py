import os

from .version import __version__
from .VerifyDeterm import VerifyDeterm
from .VerifyEnsemble import VerifyEnsemble
from .VerifyProbCSGD import VerifyProbCSGD
from .VerifyProbGaussian import VerifyProbGaussian

__title__ = "PyAnEn"
__author__ = "Weiming Hu"

Defaults = {
    'pyanen_kde_bandwidth': 0.01,
    'pyanen_kde_kernel': 'gaussian',
    'pyanen_kde_samples': 1000,
    'pyanen_kde_multiply_spread': 5,
    'pyanen_boot_confidence': 0.95,
    'pyanen_boot_repeats': 1000,
    'pyanen_boot_samples': 300,
    'pyanen_tqdm_disable': True,
    'pyanen_tqdm_leave': False,
    'pyanen_use_tensorflow_math': False,
    'pyanen_skip_nan': False
}

for k, v in Defaults.items():
    if k not in os.environ:
        os.environ[k] = str(v)
