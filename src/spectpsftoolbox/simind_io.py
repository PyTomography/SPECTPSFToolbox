from __future__ import annotations
from typing import Sequence
import numpy as np
import os
import re
import torch
from pathlib import Path

relation_dict = {'unsignedinteger': 'int',
                 'shortfloat': 'float',
                 'int': 'int'}

def get_header_value(
    list_of_attributes: list[str],
    header: str,
    dtype: type = np.float32,
    split_substr = ':=',
    split_idx = -1,
    return_all = False
    ) -> float|str|int:
    """Finds the first entry in an Interfile with the string ``header``

    Args:
        list_of_attributes (list[str]): Simind data file, as a list of lines.
        header (str): The header looked for
        dtype (type, optional): The data type to be returned corresponding to the value of the header. Defaults to np.float32.

    Returns:
        float|str|int: The value corresponding to the header (header).
    """
    header = header.replace('[', '\[').replace(']','\]').replace('(', '\(').replace(')', '\)')
    y = np.vectorize(lambda y, x: bool(re.compile(x).search(y)))
    selection = y(list_of_attributes, header).astype(bool)
    lines = list_of_attributes[selection]
    if len(lines)==0:
        return False
    values = []
    for i, line in enumerate(lines):
        if dtype == np.float32:
            values.append(np.float32(line.replace('\n', '').split(split_substr)[split_idx]))
        elif dtype == str:
            values.append(line.replace('\n', '').split(split_substr)[split_idx].replace(' ', ''))
        elif dtype == int:
            values.append(int(line.replace('\n', '').split(split_substr)[split_idx].replace(' ', '')))
        if not(return_all):
            return values[0]
    return values

def get_projections_from_single_file(headerfile: str):
    """Gets projection data from a SIMIND header file.

    Args:
        headerfile (str): Path to the header file
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.

    Returns:
        (torch.Tensor[1, Ltheta, Lr, Lz]): Simulated SPECT projection data.
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = get_header_value(headerdata, 'total number of images', int)
    if num_proj>1:
        raise ValueError('Only one projection is supported for PSF fitting')
    proj_dim1 = get_header_value(headerdata, 'matrix size [1]', int)
    proj_dim2 = get_header_value(headerdata, 'matrix size [2]', int)
    number_format = get_header_value(headerdata, 'number format', str)
    number_format= relation_dict[number_format]
    num_bytes_per_pixel = get_header_value(headerdata, 'number of bytes per pixel', int)
    imagefile = get_header_value(headerdata, 'name of data file', str)
    dtype = eval(f'np.{number_format}{num_bytes_per_pixel*8}')
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=dtype)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))[0]
    projections = torch.tensor(projections.copy())
    return projections

def get_projections(headerfiles: str | Sequence[str], weights: float = None):
    projectionss = []
    for headerfile in headerfiles:
        projections = get_projections_from_single_file(headerfile)
        if weights is not None:
            projections *= weights
        projectionss.append(projections)
    return torch.stack(projectionss)

def get_radii(resfiles: str, device='cpu'):
    radii = []
    for resfile in resfiles:
        with open(resfile) as f:
            resdata = f.readlines()
        resdata = np.array(resdata)
        radius = float(get_header_value(resdata, 'UpperEneWindowTresh:', str).split(':')[-1])
        radii.append(radius)
    return torch.tensor(radii)

def get_meshgrid(resfiles: str, device = 'cpu'):
    with open(resfiles[0]) as f:
        resdata = f.readlines()
    resdata = np.array(resdata)
    dx = float(get_header_value(resdata, 'PixelSize  I', str).split(':')[1].split('S')[0])
    dy = float(get_header_value(resdata, 'PixelSize  J', str).split(':')[1].split('S')[0])
    Nx = int(get_header_value(resdata, 'MatrixSize I', str).split(':')[1].split('I')[0])
    Ny = int(get_header_value(resdata, 'MatrixSize J', str).split(':')[1].split('A')[0])
    x = torch.arange(-(Nx-1)/2, (Nx+1)/2, 1).to(device).to(torch.float32) * dx
    y = torch.arange(-(Ny-1)/2, (Ny+1)/2, 1).to(device).to(torch.float32) * dy
    xv, yv = torch.meshgrid(x, y, indexing='xy')
    return xv, yv
    