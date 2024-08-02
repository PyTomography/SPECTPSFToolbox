from __future__ import annotations
from typing import Callable, Sequence
import torch
from torch.nn.functional import conv1d, conv2d
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from torch.nn.functional import grid_sample
from fft_conv_pytorch import fft_conv
from spectpsftoolbox.utils import pad_object, unpad_object
from spectpsftoolbox.kernel1d import GaussianKernel1D, Kernel1D
from kernel2d import Kernel2D
import dill

class Operator:
    """Base class for operators; operators are used to apply linear shift invariant operations to a sequence of 2D images.
    """
    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the operator on the input. The meshgrid xv and yv is used to compute the kernel size; it is assumed that the spacing in xv and yv is the same as that in input. The output is multiplied by the area of a pixel in the meshgrid.

        Args:
            input (torch.Tensor[Ld,Li,Lj]): Input 3D map to be operated on
            xv (torch.Tensor[Lx,Ly]): Meshgrid x coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld,Li,Lj]: Output of the operator
        """
        return input * self._area(xv,yv)
    def _area(self, xv: torch.Tensor, yv: torch.Tensor) -> float:
        """Compute pixel volume in meshgrid

        Args:
            xv (torch.Tensor): Meshgrid x coordinates
            yv (torch.Tensor): Meshgrid y coordinates

        Returns:
            float: Are
        """
        return (xv[0,1]-xv[0,0])*(yv[1,0]-yv[0,0])
    def set_device(self, device: str):
        """Sets the device of all parameters in the operator

        Args:
            device (str): Device to set parameters to
        """
        for p in self.params:
            p.data = p.data.to(device)
    def detach(self):
        """Detaches all parameters from autograd.
        """
        for p in self.params:
            p.detach_()
    def set_requires_grad(self):
        """Sets all parameters to require grad
        """
        for p in self.params:
            p.requires_grad_(True)
    def save(self, path: str):
        """Saves the operator

        Args:
            path (str): Path where to save the operator
        """
        self.set_device('cpu')
        self.detach()
        dill.dump(self, open(path, 'wb'))
    def normalization_constant(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Computes the normalization constant of the operator

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        ...
    def normalize(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes the input by the normalization constant. This ensures that the operator maintains the total sum of the input at each source-detector distance.

        Args:
            input (torch.Tensor[Ld,Li,Lj]): Input to be normalized
            xv (torch.Tensor[Lx,Ly]): Meshgrid x coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld,Li,Lj]: Normalized input
        """
        return input / self.normalization_constant(xv, yv, a)
    def __add__(self, other: Operator) -> Operator:
        """Implementation of addition to allow for adding operators. Addition of two operators yields a new operator that corresponds to the sum of two linear operators

        Args:
            other (Operator): Operator to add

        Returns:
            Operator: New operator corresponding to the sum of the two operators
        """
        def combined_operator(x, *args, **kwargs):
            return self(x, *args, **kwargs) + other(x, *args, **kwargs)
        return CombinedOperator(combined_operator, [self, other], type='additive')
    def __mul__(self, other: Operator) -> Operator:
        """Implementation of multiplication to allow for multiplying operators. Multiplication of two operators yields a new operator that corresponds to the composition of the two operators

        Args:
            other (Operator): Operator to use in composition

        Returns:
            Operator: Composed operators
        """
        def combined_operator(x, *args, **kwargs):
            return self(other(x, *args, **kwargs), *args, **kwargs)
        return CombinedOperator(combined_operator, [self,other], type='sequential')
    
class CombinedOperator(Operator):
    """Operator that has been constructed using two other operators

    Args:
        func (Callable): Function that specifies how the two operators are combined
        operators (Sequence[Operator]): Sequence of operators
        type (str): Type of operator: either 'sequential' or 'additive'
    """
    def __init__(
        self,
        func: Callable,
        operators: Sequence[Operator],
        type: str
    ) -> None:
        self.params = [*operators[0].params, *operators[1].params]
        self.func = func
        self.type = type
        self.operators = operators
        
    def set_device(self, device: str) -> None:
        """Sets the device of all the parameters in the composed operator

        Args:
            device (str): Device to set parameters to
        """
        for operator in self.operators:
            operator.set_device(device)
            
    def detach(self) -> None:
        """Detaches all parameters of the composed operator
        """
        for operator in self.operators:
            operator.detach()
        
    def normalization_constant(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Computes the normalization constant of the combined operator using the normalization constants of its components

        Args:
            xv (torch.Tensor): Meshgrid x coordinates
            yv (torch.Tensor): Meshgrid y coordinates
            a (torch.Tensor): Source-detector distances

        Returns:
            torch.Tensor: Normalization constant
        """
        if self.type=='additive':
            return self.operators[0].normalization_constant(xv, yv, a) + self.operators[1].normalization_constant(xv, yv, a)
        else:
            return self.operators[0].normalization_constant(xv, yv, a) * self.operators[1].normalization_constant(xv, yv, a)

    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        """Computes the output of the combined operator

        Args:
            input (torch.Tensor[Ld,Li,Lj]): Input to the operator
            xv (torch.Tensor[Lx,Ly]): Meshgrid x coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y coordinates
            a (torch.Tensor[Ld]): Source-detector distances
            normalize (bool, optional): Whether to normalize the output. Defaults to False.

        Returns:
            torch.Tensor[Ld,Li,Lj]: Output of the operator
        """
        if normalize:
            return self.func(input, xv, yv, a) / self.normalization_constant(xv, yv, a)
        else:
            return self.func(input,xv, yv, a)
    
class Rotate1DConvOperator(Operator):
    """Operator that functions by rotating the input by a number of angles and applying a 1D convolution at each angle

    Args:
        kernel1D (Kernel1D): 1D kernel to apply at each rotation angle
        N_angles (int): Number of angles to convolve at. Evenly distributes these angles between 0 and 180 degrees (2 angles would be 0, 90 degrees)
        additive (bool, optional): Use in additive mode; in this case, the initial input is used at each rotation angle. If False, then output from each previous angle is used in succeeding angles. Defaults to False.
        use_fft_conv (bool, optional): Whether or not to use FFT based convolution. Defaults to False.
        rot (float, optional): Initial angle offset. Defaults to 0.
    """
    def __init__(
        self,
        kernel1D: Kernel1D,
        N_angles: int,
        additive: bool = False,
        use_fft_conv: bool = False,
        rot: float = 0
    ) -> None:
        self.params = kernel1D.params
        self.kernel1D = kernel1D
        self.N_angles = N_angles
        self.angles = [180*i/N_angles + rot for i in range(N_angles)]
        self.additive = additive
        self.angle_delta = 1e-4
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input: torch.Tensor) -> torch.Tensor:
        """Applies convolution to the input

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Convolved input tensor
        """
        if self.use_fft_conv:
            return fft_conv(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        else:
            return conv1d(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        if self.additive:
            return (self.N_angles*self.kernel.sum(dim=-1)).unsqueeze(-1) * torch.sqrt(self._area(xv,yv))
        else:
            return ((self.kernel.sum(dim=-1))**self.N_angles).unsqueeze(-1) * self._area(xv,yv)
    
    def _rotate(self, input: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotates the input at the desired angle

        Args:
            input (torch.Tensor): Input tensor
            angle (float): Angle to rotate by

        Returns:
            torch.Tensor: Rotated input
        """
        if abs(angle)<self.angle_delta:
            return input
        elif abs(angle%90)<self.angle_delta:
            return torch.rot90(input, int(angle//90), dims=[1,2])
        else:
            return rotate(input, angle, interpolation=InterpolationMode.BILINEAR)
            
    def _apply_additive(self, input: torch.Tensor) -> torch.Tensor:
        """Applies the operator in additive mode

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor, which is rotated + convolved input tensor
        """
        output = 0
        for angle in self.angles:
            output_i = self._rotate(input, angle)
            output_i = output_i.swapaxes(0,1)
            output_i = self._conv(output_i)
            output_i = output_i.swapaxes(0,1)
            output_i = self._rotate(output_i, -angle)
            output += output_i
        return output
    
    def _apply_regular(self, input: torch.Tensor) -> torch.Tensor:
        """Applies operator in non-additive mode

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor, which is rotated + convolved input tensor
        """
        for angle in self.angles:
            input = self._rotate(input, angle)
            input = input.swapaxes(0,1)
            input = self._conv(input)
            input = input.swapaxes(0,1)
            input = self._rotate(input, angle)
        return input 
        
    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        input = pad_object(input)
        # Get padded kernel shape
        dx = xv[0,1] - xv[0,0]
        Nx_padded = input.shape[-1]
        if Nx_padded%2==0: Nx_padded +=1 # kernel must be odd
        x_padded = torch.arange(-(Nx_padded-1)/2, (Nx_padded+1)/2, 1).to(input.device) * dx
        self.kernel = self.kernel1D(x_padded,a,normalize=False).unsqueeze(1)
        # Apply operations
        if self.additive:
            input = self._apply_additive(input)
        else:
            input = self._apply_regular(input)
        input = unpad_object(input)
        if normalize: input = self.normalize(input, xv, yv, a)
        if self.additive:
            return input * torch.sqrt(self._area(xv,yv))
        else:
            return input * self._area(xv,yv)
    
class RotateSeperable2DConvOperator(Operator):
    """Operator that applies rotations followed by convolutions with two perpendicular 1D kernels (x/y) at each angle

    Args:
        kernel1D (Kernel1D): Kernel1D to use for convolution
        N_angles (int): Number of angles to rotate at
        additive (bool, optional): Use in additive mode; in this case, the initial input is used at each rotation angle. If False, then output from each previous angle is used in succeeding angles. Defaults to False.
        use_fft_conv (bool, optional): Whether or not to use FFT based convoltution. Defaults to False.
        rot (float, optional): Initial rotation angle. Defaults to 0.
    """
    def __init__(
        self,
        kernel1D: Kernel1D,
        N_angles: int,
        additive: bool = False,
        use_fft_conv: bool = False,
        rot: float = 0
    ) -> None:
        self.params = kernel1D.params
        self.kernel1D = kernel1D
        self.N_angles = N_angles
        self.angles = [90*i/N_angles + rot for i in range(N_angles)]
        self.additive = additive
        self.angle_delta = 1e-4
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input: torch.Tensor) -> torch.Tensor:
        """Applies convolution

        Args:
            input (torch.Tensor): Input tensor to convole

        Returns:
            torch.Tensor: Convolved input tensor
        """
        if self.use_fft_conv:
            return fft_conv(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        else:
            return conv1d(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(self, xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # """Uses recursive docstring"""
        dx = xv[0,1] - xv[0,0]
        if self.additive:
            return (self.N_angles*self.kernel.sum(dim=-1)**2).unsqueeze(-1) * self._area(xv,yv)
        else:
            return ((self.kernel.sum(dim=-1)**2)**self.N_angles).unsqueeze(-1) * self._area(xv,yv)
    
    def _rotate(self, input: torch.Tensor, angle: float) -> torch.Tensor:
        """Applies rotation to input tensor

        Args:
            input (torch.Tensor): Input tensor to be rotated
            angle (float): Rotation angle

        Returns:
            torch.Tensor: Rotated input tensor
        """
        if abs(angle)<self.angle_delta:
            return input
        elif abs(angle%90)<self.angle_delta:
            return torch.rot90(input, int(angle//90), dims=[1,2])
        else:
            return rotate(input, angle, interpolation=InterpolationMode.BILINEAR)
            
    def _apply_additive(self, input: torch.Tensor) -> torch.Tensor:
        """Applies operator in additive mode

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        output = 0
        for angle in self.angles:
            output_i = self._rotate(input, angle)
            output_i = output_i.swapaxes(0,1)
            # Perform 2D conv
            output_i = self._conv(output_i)
            output_i = output_i.swapaxes(0,-1)
            output_i = self._conv(output_i)
            output_i = output_i.swapaxes(0,-1)
            # ----
            output_i = output_i.swapaxes(0,1)
            output_i = self._rotate(output_i, -angle)
            output += output_i
        return output
    
    def _apply_regular(self, input: torch.Tensor) -> torch.Tensor:
        """Applies operator in non-additive mode

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for angle in self.angles:
            input = self._rotate(input, angle)
            input = input.swapaxes(0,1)
            # Perform 2D conv
            input = self._conv(input)
            input = input.swapaxes(0,-1)
            input = self._conv(input)
            input = input.swapaxes(0,-1)
            # ----
            input = input.swapaxes(0,1)
            input = self._rotate(input, -angle)
        return input
        
    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        self.kernel = self.kernel1D(xv[0],a,normalize=False).unsqueeze(1) # always false
        input = pad_object(input)
        if self.additive:
            input = self._apply_additive(input)
        else:
            input = self._apply_regular(input)
        input = unpad_object(input)
        if normalize: input = self.normalize(input, xv, yv, a)
        return input * self._area(xv,yv)

class Kernel2DOperator(Operator):
    """Operator built using a general 2D kernel; the output of this operator is 2D convolution with the Kernel2D instance

    Args:
        kernel2D (Kernel2D): Kernel2D instance used for obtaining the generic 2D kernel
        use_fft_conv (bool, optional): Whether or not to use FFT based convolution. Defaults to False.
    """
    def __init__(
        self,
        kernel2D: Kernel2D,
        use_fft_conv: bool = False,
    ) -> None:
        self.params = kernel2D.params
        self.kernel2D = kernel2D
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input: torch.Tensor) -> torch.Tensor:
        """Applies convolution to the input

        Args:
            input (torch.Tensor): Input

        Returns:
            torch.Tensor: Output
        """
        if self.use_fft_conv:
            return fft_conv(input, self.kernel.unsqueeze(1), padding='same', groups=self.kernel.shape[0])
        else:
            return conv2d(input, self.kernel.unsqueeze(1), padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        return self.kernel2D.normalization_constant(xv, yv, a) * self._area(xv,yv)

    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        self.kernel = self.kernel2D(xv,yv,a)
        if normalize:
            self.kernel = self.kernel / self.normalization_constant(xv, yv, a)
        return self._conv(input) * self._area(xv,yv)
    
    
class NearestKernelOperator(Operator):
    """Operator that uses a set of PSFs and distances to compute the output of the operator. The PSF is obtained by selecting the nearest PSF to each distance provided in __call__ so that each plane in input is convolved with the appropriate kernel.

    Args:
        psf_data (torch.Tensor[LD,LX,LY]): Provided PSF data
        distances (torch.Tensor[LD]): Source-detector distance for each PSF
        dr0 (float): Spacing in the PSF data
        use_fft_conv (bool, optional): Whether or not to use FFT based convolutions. Defaults to True.
        grid_sample_mode (str, optional): How to sample the PSF when the input spacing is not the same as the PSF. Defaults to 'bilinear'.
    """
    def __init__(
        self,
        psf_data: torch.Tensor,
        distances: torch.Tensor,
        dr0: float,
        use_fft_conv: bool = True,
        grid_sample_mode: str = 'bilinear'
    ) -> None:
        self.psf_data = psf_data
        self.Nx0 = psf_data.shape[1]
        self.Ny0 = psf_data.shape[2]
        self.distances_original = distances
        self.dr0 = dr0
        self.use_fft_conv = use_fft_conv
        self.params = []
        self.grid_sample_mode = grid_sample_mode
        
    def set_device(self, device: str) -> None:
        # """Uses recursive docstring"""
        self.psf_data = self.psf_data.to(device)
        self.distances_original = self.distances_original.to(device)
        
    def _conv(self, input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Performs convolution on the input data

        Args:
            input (torch.Tensor): Input data
            kernel (torch.Tensor): Kernel to convolve with

        Returns:
            torch.Tensor: Convolved input data
        """
        groups = input.shape[0]
        if self.use_fft_conv:
            return fft_conv(input.unsqueeze(0), kernel.unsqueeze(1), padding='same', groups=groups).squeeze()
        else:
            return conv2d(input.unsqueeze(0), kernel.unsqueeze(1), padding='same', groups=groups).squeeze()
    
    def _get_nearest_distance_idxs(self, distances: torch.Tensor) -> torch.Tensor:
        """Obtains the indices of the nearest PSF to each distance

        Args:
            distances (torch.Tensor): Distances to find the nearest PSF for

        Returns:
            torch.Tensor: Array of indices of the nearest PSF
        """
        differences = torch.abs(distances[:, None] - self.distances_original)
        indices = torch.argmin(differences, dim=1)
        return self.psf_data[indices]
    
    def _get_kernel(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Obtains the kernel by sampling the nearest PSF at the appropriate location

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld,Lx,Ly]: Kernel obtained by sampling the nearest PSF
        """
        dx = xv[0, 1] - xv[0, 0]
        psf = self._get_nearest_distance_idxs(a)
        grid = torch.stack([
            2*xv/(self.Nx0 * self.dr0[0]),
            2*yv/(self.Ny0 * self.dr0[1])],
            dim=-1).unsqueeze(0).repeat(a.shape[0], 1, 1, 1)
        return (dx/self.dr0[0])**2 * grid_sample(psf.unsqueeze(1), grid, align_corners=False, mode=self.grid_sample_mode).squeeze()
    
    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        kernel = self._get_kernel(xv, yv, a)
        return self._conv(input, kernel).squeeze() 
    
# TODO: Make subclass of RotateSeperable2DConvOperator with one angle and Gaussian kernel
class GaussianOperator(Operator):
    """Gaussian operator; works by convolving the input with two perpendicular 1D kernels. This is implemented seperately from the Kernel2DOperator since it is more efficient to convolve with two 1D kernels than a 2D kernel.

    Args:
        amplitude_fn (Callable): Amplitude function for 1D Gaussian kernel
        sigma_fn (Callable): Scale function for 1D Gaussian kernel
        amplitude_params (torch.Tensor): Amplitude hyperparameters
        sigma_params (torch.Tensor): Scaling hyperparameters
        a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
        a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
        use_fft_conv (bool, optional): Whether or not to use FFT based convolution. Defaults to False.
    """
    def __init__(
        self,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        a_min: float = -torch.inf,
        a_max: float = torch.inf,
        use_fft_conv: bool = False,
    ) -> None:
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        amplitude_fn1D = lambda a, bs: torch.sqrt(torch.abs(amplitude_fn(a, bs)))
        self.kernel1D = GaussianKernel1D(amplitude_fn1D, sigma_fn, amplitude_params, sigma_params, a_min, a_max)
        self.params = self.kernel1D.params
        self.a_min = a_min
        self.a_max = a_max
        self.use_fft_conv = use_fft_conv
        
    def normalization_constant(
        self,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        return self.kernel1D.normalization_constant(xv[0], a).unsqueeze(1)**2 * self._area(xv,yv)
        
    def _conv(self, input: torch.Tensor, kernel1D: torch.Tensor) -> torch.Tensor:
        """Performs convolution on input

        Args:
            input (torch.Tensor): Input tensor
            kernel1D (torch.Tensor): Gaussian 1D kernel

        Returns:
            torch.Tensor: Output convolved tensor
        """
        if self.use_fft_conv:
            return fft_conv(input, kernel1D, padding='same', groups=kernel1D.shape[0])
        else:
            return conv1d(input, kernel1D, padding='same', groups=kernel1D.shape[0])
        
    def __call__(
        self,
        input: torch.Tensor,
        xv: torch.Tensor,
        yv: torch.Tensor,
        a: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        # """Uses recursive docstring"""
        x = xv[0]
        kernel = self.kernel1D(x,a).unsqueeze(1)
        input = input.swapaxes(0,1) # x needs to be channel index
        for i in [0,2]:
            input = input.swapaxes(i,2)
            input = self._conv(input, kernel)
            input= input.swapaxes(i,2)
        input = input.swapaxes(0,1)
        if normalize: input = self.normalize(input, xv, yv, a)
        return input * self._area(xv,yv)