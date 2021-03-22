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

import numpy as np
import properscoring as ps


def crps(forecasted, observed, average_axis=None, ensemble_axis=-1, **kwargs):
    """
    Calculate CRPS
    :param forecasted: Ensemble forecast array
    :param observed: Observation array
    :param average_axis: The axis to average after collapsing the ensemble axis
    :param ensemble_axis: The ensemble axis. `None` for no averaging; `'all'` for averaging all axes.
    :param kwargs: extra arguments for `properscoring.crps_ensemble`
    :return: CRPS array
    """
    stats = ps.crps_ensemble(observed, forecasted, axis=ensemble_axis, **kwargs)

    if average_axis is not None:
        if average_axis is 'all':
            stats = np.nanmean(stats, axis=None)
        else:
            stats = np.nanmean(stats, axis=average_axis)

    return stats


def bias(forecasted, observed, average_axis=None):
    """
    Calculate bias
    :param forecasted: Forecast array
    :param observed: Observation array
    :param average_axis: The axis to average.  `None` for no averaging; `'all'` for averaging all axes.
    :return: Bias array
    """
    stats = forecasted - observed

    if average_axis is not None:
        if average_axis is 'all':
            stats = np.nanmean(stats, axis=None)
        else:
            stats = np.nanmean(stats, axis=average_axis)

    return stats


def rmse(forecasted, observed, average_axis=None):
    """
    Calculate RMSE
    :param forecasted: Forecast array
    :param observed: Observation array
    :param average_axis: The axis to average.  `None` for averaging all axes.
    :return: RMSE array
    """
    stats = np.sqrt(np.nanmean(np.power(forecasted - observed, 2), axis=average_axis))

    return stats


def mae(forecasted, observed, average_axis=None):
    """
    Calculate MAE
    :param forecasted: Forecast array
    :param observed: Observation array
    :param average_axis: The axis to average.  `None` for averaging all axes.
    :return: RMSE array
    """
    stats = np.nanmean(np.abs(forecasted - observed), axis=average_axis)

    return stats
