from __future__ import annotations
from typing import Callable
import torch
import numpy as np
from functools import partial
from torch.nn.functional import conv1d, conv2d, grid_sample
from torchvision.transforms.functional import rotate
from torchquad import Simpson
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Kernel2D:
    def __init__(self, a_min: float, a_max: float) -> None:
        """Parent class for 2D kernels. All children class should inherit from this class and implement the __call__ and normalization_constant methods.

        Args:
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. 
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value.
        """
        self.a_min = a_min
        self.a_max = a_max
    def normalization_constant(self, a: torch.Tensor) -> torch.Tensor:
        """Computes the normalization constant for the kernel at each source-detector distance a. This method should be implemented in the child class.

        Args:
            a (torch.Tensor[Ld]): Source detector distances

        Returns:
            torch.Tensor[Ld]: Normalization at each source-detector distance
        """
        ...
    def __call__(xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Computes the kernel value at each point in the meshgrid defined by xv and yv for each source-detector distance a. This method should be implemented in the child class.

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x-coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y-coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld,Lx,Ly]: Kernel values at each point in the meshgrid for each source-detector distance
        """
        ...
        
class FunctionalKernel2D(Kernel2D):
    def __init__(
        self,
        kernel_fn: Callable,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        a_min: float = -torch.inf,
        a_max: float = torch.inf
    ) -> None:
        """2D kernel where the kernel is specified explicitly given a function of x and y. The kernel is evaluated as f(x,y) = amplitude(a,b) * k(x/sigma(a,b),y/sigma(a,b)) where amplitude and sigma are functions of the source-detector distance a and additional hyperparameters b.

        Args:
            kernel_fn (Callable): Kernel function k(x,y)
            amplitude_fn (Callable): Amplitude function amplitude(a,b)
            sigma_fn (Callable): Scaling function sigma(a,b)
            amplitude_params (torch.Tensor): Amplitude hyperparameters b
            sigma_params (torch.Tensor): Scaling hyperparameters b
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
        """
        super(FunctionalKernel2D, self).__init__(a_min, a_max)
        self.kernel_fn = kernel_fn
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.all_parameters = [amplitude_params, sigma_params]
        self._compute_norm_via_integral()
        
    def _compute_norm_via_integral(self) -> None:
        """Compute the normalization constant by integrating k(x,y) from -infinity to infinity. To do this, a variable transformation is used to convert the integral to a definite integral over the range [-pi/2, pi/2]. The definite integral is computed using the Simpson's rule.
        """
        # Convert to definite integral
        kernel_fn_definite = lambda t: self.kernel_fn(torch.tan(t[:,0]), torch.tan(t[:,1])) * torch.cos(t[:,0])**(-2) * torch.cos(t[:,1])**(-2)
        # Should be good for most simple function cases
        self.kernel_fn_norm = Simpson().integrate(kernel_fn_definite, dim=2, N=1001, integration_domain=[[-torch.pi/2, torch.pi/2], [-torch.pi/2, torch.pi/2]])
        
    def normalization_constant(self, xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Obtains the normalization constant for the 2D kernel

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x-coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y-coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source-detector distance
        """
        return self.kernel_fn_norm*self.amplitude_fn(a, self.amplitude_params)*self.sigma_fn(a, self.sigma_params)**2 / (xv[0,1]-xv[0,0]) / (yv[1,0] - yv[0,0])

    def __call__(self, xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Computes the kernel at each source detector distance

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x-coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y-coordinates
            a (torch.Tensor[Ld]): Source-detector distances
            normalize (bool, optional): Whether or not to normalize the output of the kernel. Defaults to False.

        Returns:
            torch.Tensor[Ld,Lx,Ly]: Kernel at each source-detector distance
        """
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        N = 1 if normalize is False else self.normalization_constant(xv, yv, a)
        return self.amplitude_fn(a, self.amplitude_params) * self.kernel_fn(xv/self.sigma_fn(a, self.sigma_params), yv/self.sigma_fn(a, self.sigma_params)) / N

class NGonKernel2D(Kernel2D):
    def __init__(
        self,
        N_sides: int,
        Nx: int,
        collimator_width: float,
        amplitude_fn: Callable,
        sigma_fn: Callable,
        amplitude_params: torch.Tensor,
        sigma_params: torch.Tensor,
        a_min = -torch.inf,
        a_max = torch.inf,
        rot: float = 0  
    ) -> None:
        """Implementation of the arbitrary polygon kernel. This kernel is composed of a polygon shape convolved with itself, which is shown to be the true geometric component of the SPECT PSF when averaged over random collimator movement to get a linear shift invariant approximation. The kernel is computed as f(x,y) = amplitude(a,b) * k(x/sigma(a,b),y/sigma(a,b)) where k(x,y) is the convolved polygon shape.

        Args:
            N_sides (int): Number of sides of the polygon. Currently only supports even side lengths
            Nx (int): Number of voxels to use for constructing the polygon (seperate from any meshgrid stuff done later on)
            collimator_width (float): Width of the polygon (from flat edge to flat edge)
            amplitude_fn (Callable): Amplitude function amplitude(a,b)
            sigma_fn (Callable): Scaling function sigma(a,b)
            amplitude_params (torch.Tensor): Amplitude hyperparameters b
            sigma_params (torch.Tensor): Scaling hyperparameters b
            a_min (float, optional): Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to -torch.inf.
            a_max (float, optional):Minimum source-detector distance for the kernel; any distance values passed to __call__ below this value will be clamped to this value. Defaults to torch.inf.
            rot (float, optional): Initial rotation of the polygon flat side. Defaults to 0 (first flat side aligned with +y axis).
        """
        self.N_sides = N_sides
        self.Nx = Nx
        self.N_voxels_to_face = int(np.floor(Nx/6 * np.cos(np.pi/self.N_sides)))
        self.collimator_width_voxels = 2 * self.N_voxels_to_face
        self.pixel_size = collimator_width / self.collimator_width_voxels
        self.collimator_width = collimator_width
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.rot = rot
        self._compute_convolved_polygon()
        self.a_min = a_min
        self.a_max = a_max
        self.params = [amplitude_params, sigma_params]
        
         
    def _compute_convolved_polygon(self):
        """Computes the convolved polygon
        """
        # Create
        x = torch.zeros((self.Nx,self.Nx)).to(device)
        x[:(self.Nx-1)//2 + self.N_voxels_to_face] = 1
        polygon = []
        for i in range(self.N_sides):
            polygon.append(rotate(x.unsqueeze(0), 360*i/self.N_sides+self.rot).squeeze())
        polygon = torch.stack(polygon).prod(dim=0)
        convolved_polygon = conv2d(polygon.unsqueeze(0).unsqueeze(0), polygon.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
        self.convolved_polygon = convolved_polygon / convolved_polygon.max()
        self.convolved_polygon_sum = self.convolved_polygon.sum()
        
    def normalization_constant(self, xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Computes the normalization constant for the kernel at each source-detector distance a.

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x-coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y-coordinates
            a (torch.Tensor[Ld]): Source-detector distances

        Returns:
            torch.Tensor[Ld]: Normalization constant at each source detector distance.
        """
        if self.grid_max<1:
            # For cases where the polygon kernel exceeds the boundary
            dx = xv[0,1] - xv[0,0]
            dy = yv[1,0] - yv[0,0]
            a = torch.clamp(a, self.a_min, self.a_max)
            a = a.reshape(-1,1,1)
            return self.amplitude_fn(a, self.amplitude_params) * self.pixel_size**2 * self.sigma_fn(a, self.sigma_params)**2 * self.convolved_polygon_sum  / dx / dy
        else:
            # This is called nearly 100% of the time
            return self.kernel.sum(dim=(1,2)).reshape(-1,1,1)
    
    def __call__(self, xv: torch.Tensor, yv: torch.Tensor, a: torch.Tensor, normalize: bool = False):
        """Computes the kernel at each source detector distance

        Args:
            xv (torch.Tensor[Lx,Ly]): Meshgrid x-coordinates
            yv (torch.Tensor[Lx,Ly]): Meshgrid y-coordinates
            a (torch.Tensor[Ld]): Source-detector distances
            normalize (bool, optional): Whether or not to normalize the output of the kernel. Defaults to False.

        Returns:
            torch.Tensor[Ld,Lx,Ly]: Kernel at each source-detector distance
        """
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        sigma = self.sigma_fn(a, self.sigma_params)
        grid = torch.stack([
            2*xv/(self.Nx*self.pixel_size*sigma),
            2*yv/(self.Nx*self.pixel_size*sigma)],
            dim=-1)
        self.grid_max = grid.max()
        amplitude = self.amplitude_fn(a, self.amplitude_params).reshape(a.shape[0],1,1)
        self.kernel = amplitude * grid_sample(self.convolved_polygon.unsqueeze(0).unsqueeze(0).repeat(a.shape[0],1,1,1), grid, mode = 'bilinear', align_corners=False)[:,0]
        if normalize:
            self.kernel = self.kernel / self.normalization_constant(xv, yv, a)
        return self.kernel
            
            
            
