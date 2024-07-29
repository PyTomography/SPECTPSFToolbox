from __future__ import annotations
from typing import Callable
import torch
import numpy as np
from torch.nn.functional import grid_sample
from torchquad import Simpson

class ArbitraryKernel1D:
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
        """Kernel defined using an arbitrary function 

        Args:
            kernel (torch.Tensor): _description_
            amplitude_fn (Callable): _description_
            sigma_fn (Callable): _description_
            amplitude_params (torch.Tensor): _description_
            sigma_params (torch.Tensor): _description_
            dx0 (float): _description_
            a_min (float, optional): _description_. Defaults to -torch.inf.
            a_max (float, optional): _description_. Defaults to torch.inf.
            grid_sample_mode (str, optional): _description_. Defaults to 'bilinear'.
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
        
    def normalization_constant(self,x,a):
        return self.amplitude_fn(a, self.amplitude_params) * torch.abs(self.kernel.sum()) * self.sigma_fn(a, self.sigma_params) * self.dx0 / (x[1]-x[0])
        
    def __call__(self, x, a, normalize=False):
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
    
class FunctionKernel1D:
    def __init__(self, kernel_fn, amplitude_fn, sigma_fn, amplitude_params, sigma_params, a_min=-torch.inf, a_max=torch.inf, kernel_fn_norm=None):
        self.kernel_fn = kernel_fn
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.params = [self.amplitude_params, self.sigma_params]
        self.a_min = a_min
        self.a_max = a_max
        self.compute_norm_via_integral()
        
    def compute_norm_via_integral(self):
        # Convert to definite integral
        kernel_fn_definite = lambda t: self.kernel_fn(torch.tan(t)) * torch.cos(t)**(-2)
        # Should be good for most simple function cases
        self.kernel_fn_norm = Simpson().integrate(kernel_fn_definite, dim=1, N=1001, integration_domain=[[-torch.pi/2, torch.pi/2]])
    
    def normalization_constant(self,x,a):
        if self.kernel_fn_norm is not None:
            return self.kernel_fn_norm*self.amplitude_fn(a, self.amplitude_params)*self.sigma_fn(a, self.sigma_params) / (x[1]-x[0])
        else:
            raise NotImplementedError('kernel_fn_norm not provided for FunctionKernel1D')    
    
    def __call__(self, x, a, normalize=False):
        a = a.reshape(-1,1)
        a = torch.clamp(a, self.a_min, self.a_max)
        sigma = self.sigma_fn(a, self.sigma_params)
        amplitude = self.amplitude_fn(a, self.amplitude_params)
        kernel = amplitude*self.kernel_fn(x/sigma)
        if normalize:
            kernel = kernel / self.normalization_constant(x,a)
        return kernel
    
class GaussianKernel1D(FunctionKernel1D):
    def __init__(self, amplitude_fn, sigma_fn, amplitude_params, sigma_params, a_min=-torch.inf, a_max=torch.inf):
        kernel_fn = lambda x: torch.exp(-0.5*x**2)
        super(GaussianKernel1D, self).__init__(kernel_fn, amplitude_fn, sigma_fn, amplitude_params, sigma_params,a_min, a_max)
        
    def normalization_constant(self,x,a):
        dx = x[1] - x[0]
        sigma = self.sigma_fn(a, self.sigma_params)
        FOV_to_sigma_ratio = x.max() / sigma.max()
        if FOV_to_sigma_ratio > 4:
            return self(x,a).sum(dim=1).reshape(-1,1)
        else:
            a = torch.clamp(a, self.a_min, self.a_max)
            a = a.reshape(-1,1)
            return np.sqrt(2 * torch.pi) * self.amplitude_fn(a, self.amplitude_params) * self.sigma_fn(a, self.sigma_params) / dx