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
import re
import glob
import warnings

import numpy as np
import xarray as xr

from netCDF4 import Dataset


coords_dict = {'num_flts': 'FLTs',
               'num_parameters': 'ParameterNames',
               'num_test_times': 'test_times',
               'num_search_times': 'search_times',
               'num_times': 'Times'}


def _decode_name_matrix(mat):
    names = []
    
    for row in mat:
        names.append(''.join([v.decode("utf-8") for v in row]))
        
    return names


def __add_coords__(ds, dim_name='num_stations'):
    # Get current chunk number
    matched = re.findall(r'\d+', os.path.basename(ds.encoding['source']))
    assert len(matched) > 0, 'No anchor numbers found in file names as indices!'
    start = int(''.join(matched))

    # Get the total number of stations
    assert dim_name in ds.dims, 'The dimension {} does not exist in the following dataset: \n{}'.format(dim_name, ds)
    current_total = len(ds[dim_name])

    # Calculate a helper index
    station_coords = [start + index / current_total for index in range(current_total)]
    ds.coords['num_stations'] = station_coords
    
    # Deal with coordinate variables
    for key, value in coords_dict.items():
        if key in ds.dims and value in ds:
            ds.coords[key] = ds[value]

    return ds


def open_dataset(file, group=None, decode=False):

    ####################
    # Open the dataset #
    ####################

    ds = xr.open_dataset(file, group=group, decode_cf=False)

    ############################
    # Post-process the dataset #
    ############################
    
    # If num_char is present, then I need to decode names from matrices
    if 'num_chars' in ds.dims:
        for var in ds.variables:
            if 'num_chars' in ds[var].dims:
                
                if len(ds[var].dims) - 1 != 1:
                    warnings.warn('Cannot decode {}. Return immediately.'.format(var))
                    return ds
                
                names = _decode_name_matrix(ds[var].data)
                dim = [dim for dim in ds[var].dims if dim != 'num_chars'][0]
                ds[var] = ((dim), names)

    # Deal with coordinate variables
    for key, value in coords_dict.items():
        if key in ds.dims and value in ds:
            ds.coords[key] = ds[value]

    # Deal with time units
    for var in ['test_times', 'search_times', 'Times', 'num_test_times', 'num_times']:
        if var in ds:
            ds[var] = ds[var].assign_attrs(units='seconds since 1970-01-01')

    # Deal with test time name
    #
    # Analogs have num_test_times and num_search_times to distinguish the two types.
    # Rename num_test_times to num_times to be consistent with other groups.
    #
    if 'num_times' not in ds.dims and 'num_test_times' in ds.dims:
        ds = ds.rename(num_test_times='num_times')

    # Decode time related variables
    if decode:
        ds = xr.decode_cf(ds)

    return ds


def open_mfdataset(paths, group=None, parallel=True, decode=False):
    """
    Read multiple NetCDF files as an xarray Dataset.
    
    :param paths: See xarray.open_mfdataset
    :param group: See xarray.open_mfdataset
    :param parallel: See xarray.open_mfdataset
    :param decode: Whether to decode time related dimensions
    :return xarray Dataset
    """

    #########
    # Setup #
    #########

    if isinstance(paths, str):
        paths = glob.glob(os.path.expanduser(paths))
    else:
        paths = [os.path.expanduser(path) for path in paths]

    paths.sort()
    
    ####################
    # Open the dataset #
    ####################
    
    ds = xr.open_mfdataset(paths=paths, preprocess=__add_coords__, data_vars='minimal', coords='minimal',
                           compat='override', parallel=parallel, group=group, decode_cf=decode)
    
    ############################
    # Post-process the dataset #
    ############################

    # Deal with missing time variables by reading from the root group
    nc_root = Dataset(paths[0])

    for var in ['num_flts', 'num_test_times']:
        if var in ds.dims and var not in ds.coords:
            ds.coords[var] = nc_root.variables[coords_dict[var]][:].data

    nc_root.close()
    
    # Deal with missing location variables by reading from the root group
    if 'Xs' not in ds:
        with xr.open_mfdataset(paths=paths, preprocess=__add_coords__, concat_dim='num_stations',
                               data_vars='minimal', coords='minimal', compat='override',
                               parallel=parallel, decode_cf=False) as ds_root:

            ds = ds.assign(Xs=(ds_root['Xs'].dims, ds_root['Xs'].values),
                           Ys=(ds_root['Ys'].dims, ds_root['Ys'].values))

    # Deal with time units
    for var in ['test_times', 'search_times', 'Times', 'num_test_times', 'num_times']:
        if var in ds:
            ds[var] = ds[var].assign_attrs(units='seconds since 1970-01-01')

    # Deal with test time name
    #
    # Analogs have num_test_times and num_search_times to distinguish the two types.
    # Rename num_test_times to num_times to be consistent with other groups.
    #
    if 'num_times' not in ds.dims and 'num_test_times' in ds.dims:
        ds = ds.rename(num_test_times='num_times')

    # Decode time related variables
    if decode:
        ds = xr.decode_cf(ds)
    
    return ds
