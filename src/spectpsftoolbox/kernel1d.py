from __future__ import annotations
from typing import Callable
import torch
import numpy as np
from torch.nn.functional import grid_sample
from torchquad import Simpson

class Kernel1D:
    def __init__(self, a_min: float, a_max: float) -> None:
        """Super class for all implemented 1D kernels in the library. All child classes should implement the normalization_factor and __call__ methods.

        Args:
            a_min (float): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value.
            a_max (float): Maximum source-detector distance for the kernel; any distance values passed to __call__ above this value will be clamped to this value.
        """
        self.a_min = a_min
        self.a_max = a_max
    def normalization_constant(self, x: torch.Tensor,a: torch.Tensor) -> torch.Tensor:
        """Computes the normalization constant for the kernel

        Args:
            x (torch.Tensor[Lx]): Positions where the kernel is being evaluated
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the normalization constant

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        ...
    def __call__(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Returns the kernel evaluated at each source detector distance

        Args:
            x (torch.Tensor[Lx]): x values at which to evaluate the kernel
            a (torch.Tensor[Ld]): source-detector distances at which to evaluate the kernel
        Returns:
            torch.Tensor[Ld, Lx]: Kernel evaluated at each source-detector distance and x value
        """
        ...

class ArbitraryKernel1D(Kernel1D):
    def __init__(
        self,
        kernel: torch.Tensor,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        dx0: float,
        a_min: float = -torch.inf,
        a_max: float = torch.inf,
        grid_sample_mode: str = 'bilinear'
        ) -> None:
        """1D Kernel defined using an arbitrary 1D array as the kernel with spacing dx0. The kernel is evaluated as f(x) = amplitude(a,b) * kernel_interp(x/sigma(a,b)) where amplitude and sigma are functions of the source-detector distance a and additional hyperparameters b. kernel_interp is the 1D interpolation of the kernel at the provided x values.

        Args:
            kernel (torch.Tensor): 1D array that defines the kernel
            amplitude_fn (Callable): Amplitude function that depends on the source-detector distance and additional hyperparameters
            sigma_fn (Callable): Scaling function that depends on the source-detector distance and additional hyperparameters
            amplitude_params (torch.Tensor): Hyperparameters for the amplitude function
            sigma_params (torch.Tensor): Hyperparameters for the sigma function
            dx0 (float): Spacing of the 1D kernel provided in the same units as x used in __call__
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
            grid_sample_mode (str, optional): How to sample the kernel for general grids. Defaults to 'bilinear'.
        """
        self.kernel = kernel
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.params = [self.amplitude_params, self.sigma_params, self.kernel]
        self.dx0 = dx0
        self.Nx0 = kernel.shape[0]
        self.a_min = a_min
        self.a_max = a_max
        self.grid_sample_mode = grid_sample_mode
        
    def normalization_constant(self, x: torch.Tensor,a: torch.Tensor) -> torch.Tensor:
        """Computes the normalization constant for the kernel

        Args:
            x (torch.Tensor[Lx]): Positions where the kernel is being evaluated
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the normalization constant

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        return self.amplitude_fn(a, self.amplitude_params) * torch.abs(self.kernel.sum()) * self.sigma_fn(a, self.sigma_params) * self.dx0 / (x[1]-x[0])
        
    def __call__(self, x: torch.Tensor, a: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Calls the kernel function

        Args:
            x (torch.Tensor[Lx]): Positions where the kernel is being evaluated
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the kernel
            normalize (bool, optional): Whether or not to normalize the output of the kernel. Defaults to False.

        Returns:
            torch.Tensor[Ld,Lx]: Kernel evaluated at each source-detector distance and x value
        """
        a = torch.clamp(a, self.a_min, self.a_max)
        sigma = self.sigma_fn(a, self.sigma_params)
        amplitude = self.amplitude_fn(a, self.amplitude_params)
        grid = (2*x / (self.Nx0*self.dx0 * sigma.unsqueeze(1).unsqueeze(1)))
        grid = torch.stack([grid, 0*grid], dim=-1)
        kernel = torch.abs(self.kernel).reshape(1,1,-1).repeat(a.shape[0],1,1)
        kernel = grid_sample(kernel.unsqueeze(1), grid, mode=self.grid_sample_mode, align_corners=False)[:,0,0]
        kernel = amplitude.reshape(-1,1) * kernel
        if normalize:
            kernel = kernel / self.normalization_constant(x,a.reshape(-1,1))
        return kernel
    
class FunctionKernel1D(Kernel1D):
    def __init__(
        self,
        kernel_fn: Callable,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        a_min=-torch.inf,
        a_max=torch.inf,
        ) -> None:
        """Implementation of kernel1D where an explicit functional form is provided for the kernel. The kernel is evaluated as f(x) = amplitude(a,b) * k(x/sigma(a,b)) where amplitude and sigma are functions of the source-detector distance a and additional hyperparameters b.

        Args:
            kernel_fn (Callable): Kernel function k(x)
            amplitude_fn (Callable): Amplitude function amplitude(a,b)
            sigma_fn (Callable): Scaling function sigma(a,b)
            amplitude_params (torch.Tensor): Hyperparameters for the amplitude function
            sigma_params (torch.Tensor): Hyperparameters for the sigma function
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
        """
        self.kernel_fn = kernel_fn
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.params = [self.amplitude_params, self.sigma_params]
        self.a_min = a_min
        self.a_max = a_max
        self._compute_norm_via_integral()
        
    def _compute_norm_via_integral(self) -> torch.Tensor:
        """Compute the normalization constant by integrating k(x) from -infinity to infinity. To do this, a variable transformation is used to convert the integral to a definite integral over the range [-pi/2, pi/2]. The definite integral is computed using the Simpson's rule.

        Returns:
            torch.Tensor: Integral of k(x) from -infinity to infinity
        """
        # Convert to definite integral
        kernel_fn_definite = lambda t: self.kernel_fn(torch.tan(t)) * torch.cos(t)**(-2)
        # Should be good for most simple function cases
        self.kernel_fn_norm = Simpson().integrate(kernel_fn_definite, dim=1, N=1001, integration_domain=[[-torch.pi/2, torch.pi/2]])
    
    def normalization_constant(self,x: torch.Tensor,a: torch.Tensor) -> torch.Tensor:
        """Computes the normalization constant for the kernel

        Args:
            x (torch.Tensor[Lx]): Positions where the kernel is being evaluated
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the normalization constant
        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        return self.kernel_fn_norm*self.amplitude_fn(a, self.amplitude_params)*self.sigma_fn(a, self.sigma_params) / (x[1]-x[0])  
    
    def __call__(self, x: torch.Tensor, a: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Computes the kernel at each source-detector distance

        Args:
            x (torch.Tensor[Lx]): Positions where the kernel is being evaluated
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the kernel
            normalize (bool, optional): Whether or not to normalize the output. Defaults to False.

        Returns:
            torch.Tensor[Ld,Lx]: Kernel evaluated at each source-detector distance and x value
        """
        a = a.reshape(-1,1)
        a = torch.clamp(a, self.a_min, self.a_max)
        sigma = self.sigma_fn(a, self.sigma_params)
        amplitude = self.amplitude_fn(a, self.amplitude_params)
        kernel = amplitude*self.kernel_fn(x/sigma)
        if normalize:
            kernel = kernel / self.normalization_constant(x,a)
        return kernel
    
class GaussianKernel1D(FunctionKernel1D):
    def __init__(
        self,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        a_min: float = -torch.inf,
        a_max: float = torch.inf
        ) -> None:
        """Subclass of FunctionKernel1D that implements a Gaussian kernel. The kernel is evaluated as f(x) = amplitude(a,b) * exp(-0.5*x^2/sigma(a,b)^2) where amplitude and sigma are functions of the source-detector distance a and additional hyperparameters b.

        Args:
            amplitude_fn (Callable): Amplitude function amplitude(a,b)
            sigma_fn (Callable): Scaling function sigma(a,b)
            amplitude_params (torch.Tensor): Amplitude hyperparameters
            sigma_params (torch.Tensor): Sigma hyperparameters
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
        """
        kernel_fn = lambda x: torch.exp(-0.5*x**2)
        super(GaussianKernel1D, self).__init__(kernel_fn, amplitude_fn, sigma_fn, amplitude_params, sigma_params,a_min, a_max)
        
    def normalization_constant(self, x: torch.Tensor, a: torch.Tensor):
        """Computes the normalization constant for the kernel. For small FOV to sigma ratios, the normalization constant is computed using the definite integral of the kernel. For large FOV to sigma ratios, the normalization constant is computed by summing the values of the kernel; this prevents normalization errors when the Gaussian function is large compared to the pixel size.

        Args:
            x (torch.Tensor[Lx]): Values at which to compute the kernel
            a (torch.Tensor[Ld]): Source-detector distances at which to compute the normalization constant

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        dx = x[1] - x[0]
        sigma = self.sigma_fn(a, self.sigma_params)
        FOV_to_sigma_ratio = x.max() / sigma.max()
        if FOV_to_sigma_ratio > 4:
            return self(x,a).sum(dim=1).reshape(-1,1)
        else:
            a = torch.clamp(a, self.a_min, self.a_max)
            a = a.reshape(-1,1)
            return np.sqrt(2 * torch.pi) * self.amplitude_fn(a, self.amplitude_params) * self.sigma_fn(a, self.sigma_params) / dx