import os
import warnings

import numpy as np
from .IO import open_dataset


def read_station_coordinates(file, group=None, var_x='Xs', var_y='Ys'):
    """
    Read the x and y coordinates from an NetCDF file.
    
    :param file: Input NetCDF file
    :param group: The group of the NetCDF file to read. See xarray.open_mfdataset for details.
    :param var_x: Variable name for x coordinates
    :param var_y: Variable name for y coordinates
    :return (x, y) with each as an Numpy array
    """
    
    with open_dataset(file, group=group) as da:
        x = da[var_x].data
        y = da[var_y].data
    
    assert len(x) == len(y), 'The numbers of coordinates do not match!'
    
    return x, y


def write_matrix(file, mat, overwrite=False):
    """
    Write a matrix with a compatible form that can be used by PAnEn and DeepAnalogs.
    
    :param file: Output text file
    :param mat: A 2-dimensional Numpy array
    :param overwrite: Whether to overwrite an existing file
    :return True if everything is successful
    """
    
    # Sanity check
    assert len(mat.shape) == 2, 'Can only write a matrix!'
    
    if os.path.exists(file) and not overwrite:
        raise Exception('File {} exists. No overwriting.'.format(file))
        
    if np.count_nonzero(np.isnan(mat)) != 0:
        warnings.warn('Having NAN in the grid matrix is discouraged!')
    
    with open(file, 'w') as out_file:
        
        # Write dimension info
        meta_info = '[{}, {}]\n'.format(mat.shape[0], mat.shape[1])
        out_file.write(meta_info)
        
        # Write matrix
        out_file.write('(\n')
        out_file.writelines(['({}),\n'.format(', '.join(
            ['nan' if np.isnan(e) else str(e) for e in mat[i_row]])) for i_row in range(mat.shape[0])])
        out_file.write(')')
    
    return True
