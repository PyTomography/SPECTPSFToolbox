import torch
import numpy as np
from torch.nn.functional import pad

def compute_pad_size(width: int) -> int:
    """Computes the pad size required for rotation / inverse rotation so that pad + rotate + inverse rotate + unpad = original object

    Args:
        width (int): Width of input tensor (assumed to be square)

    Returns:
        int: How much to pad each side by
    """
    return int(np.ceil((np.sqrt(2)*width - width)/2)) 

def compute_pad_size_padded(width: int) -> int:
    """Given a padded tensor, computes how much padding it has had

    Args:
        width (int): Width of input padded tensor (assumed square)

    Returns:
        int: How much padding was applied to the input tensor
    """
    a = (np.sqrt(2) - 1)/2
    if width%2==0:
        width_old = int(2*np.floor((width/2)/(1+2*a)))
    else:
        width_old = int(2*np.floor(((width-1)/2)/(1+2*a)))
    return int((width-width_old)/2)

def pad_object(object: torch.Tensor, mode='constant') -> torch.Tensor:
    """Pads an input tensor so that pad + rotate + inverse rotate + unpad = original object. This is useful for rotating objects without losing information at the edges.

    Args:
        object (torch.Tensor): Object to be padded
        mode (str, optional): Mode for extrapolation beyonf out of bounds. Defaults to 'constant'.

    Returns:
        torch.Tensor: Padded object
    """
    pad_size = compute_pad_size(object.shape[-2]) 
    return pad(object, [pad_size,pad_size,pad_size,pad_size], mode=mode)

def unpad_object(object: torch.Tensor) -> torch.Tensor:
    """Given a padded object, removes the padding to return the original object

    Args:
        object (torch.Tensor): Padded object

    Returns:
        torch.Tensor: Unpadded, original object
    """
    pad_size = compute_pad_size_padded(object.shape[-2])
    return object[:,pad_size:-pad_size,pad_size:-pad_size]

def get_kernel_meshgrid(
    xv_input: torch.Tensor,
    yv_input: torch.Tensor,
    k_width: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Obtains a kernel meshgrid of given spatial width k_width (in same units as meshgrid). Enforces the kernel size is odd

    Args:
        xv_input (torch.Tensor): Meshgrid x-coordinates corresponding to the input of some operator
        yv_input (torch.Tensor): Meshgrid y-coordinates corresponding to the input of some operator
        k_width (float): Width of kernel in same units as meshgrid

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Meshgrid of kernel
    """
    dx = xv_input[0,1] - xv_input[0,0]
    dy = yv_input[1,0] - yv_input[0,0]
    x_kernel = torch.arange(0,k_width/2,dx).to(xv_input.device)
    x_kernel = torch.cat([-x_kernel.flip(dims=(0,))[:-1], x_kernel])
    y_kernel = torch.arange(0,k_width/2,dy).to(xv_input.device)
    y_kernel = torch.cat([-y_kernel.flip(dims=(0,))[:-1], y_kernel])
    xv_kernel, yv_kernel = torch.meshgrid(x_kernel, y_kernel, indexing='xy')
    return xv_kernel, yv_kernel