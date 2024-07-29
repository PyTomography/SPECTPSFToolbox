import torch
import numpy as np
from functools import partial
from torch.nn.functional import conv1d, conv2d, grid_sample
from torchvision.transforms.functional import rotate
from torchquad import Simpson
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Put "parametrized" in front of all of thsese

class Kernel2D:
    def __init__(self, a_min, a_max, dx0, dy0):
        self.a_min = a_min
        self.a_max = a_max
        self.dx0 = dx0
        self.dy0 = dy0
    def normalization_factor(self, a):
        ...
    def __call__(xv, yv, a):
        ...
        
class FunctionalKernel2D(Kernel2D):
    def __init__(
        self,
        kernel_fn,
        amplitude_fn,
        sigma_fn,
        amplitude_params,
        sigma_params,
        a_min = -torch.inf,
        a_max = torch.inf
    ):  
        super(FunctionalKernel2D, self).__init__(a_min, a_max, dx0=None, dy0=None)
        self.kernel_fn = kernel_fn
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        self.all_parameters = [amplitude_params, sigma_params]
        self.compute_norm_via_integral()
        
    def compute_norm_via_integral(self):
        # Convert to definite integral
        kernel_fn_definite = lambda t: self.kernel_fn(torch.tan(t[:,0]), torch.tan(t[:,1])) * torch.cos(t[:,0])**(-2) * torch.cos(t[:,1])**(-2)
        # Should be good for most simple function cases
        self.kernel_fn_norm = Simpson().integrate(kernel_fn_definite, dim=2, N=1001, integration_domain=[[-torch.pi/2, torch.pi/2], [-torch.pi/2, torch.pi/2]])
        
    def normalization_constant(self,xv, yv, a):
        return self.kernel_fn_norm*self.amplitude_fn(a, self.amplitude_params)*self.sigma_fn(a, self.sigma_params)**2 / (xv[0,1]-xv[0,0]) / (yv[1,0] - yv[0,0])

    def __call__(self, xv, yv, a, normalize=False):
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        N = 1 if normalize is False else self.normalization_constant(xv, yv, a)
        return self.amplitude_fn(a, self.amplitude_params) * self.kernel_fn(xv/self.sigma_fn(a, self.sigma_params), yv/self.sigma_fn(a, self.sigma_params)) / N
        
class GeneralizedGaussianKernel2D(Kernel2D):
    def __init__(
        self,
        amplitude_fn,
        mu_fn,
        sigma_fn,
        alpha_fn,
        amplitude_params,
        mu_params,
        sigma_params,
        alpha_params,
        dx0,
        dy0,
        a_min = -torch.inf,
        a_max = torch.inf
    ):  
        super(GeneralizedGaussianKernel2D, self).__init__(a_min, a_max, dx0, dy0)
        self.amplitude_fn = amplitude_fn
        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn
        self.alpha_fn = alpha_fn
        self.amplitude_params = amplitude_params
        self.mu_params = mu_params
        self.sigma_params = sigma_params
        self.alpha_params = alpha_params
        self.all_parameters = [amplitude_params, sigma_params, alpha_params]
        
    def __call__(self, xv, yv, a, normalize=False):
        N = 1
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        return self.amplitude_fn(a, self.amplitude_params) * torch.exp(
            -((xv - self.mu_fn(a, self.mu_params))**2 + (yv - self.mu_fn(a, self.mu_params))**2)**self.alpha_fn(a, self.alpha_params) / (2 * self.sigma_fn(a, self.sigma_params)**2)
        ) / N
        
class ArbitraryKernel2D(Kernel2D):
    def __init__(
        self,
        kernel,
        amplitude_fn,
        sigma_fn,
        amplitude_params,
        sigma_params,
        dx0,
        dy0,
        a_min=-torch.inf,
        a_max=torch.inf,
        kernel_trainable=True
    ):
        super(ArbitraryKernel2D, self).__init__(a_min, a_max, dx0, dy0)
        self.kernel = kernel
        self.amplitude_fn = amplitude_fn
        self.sigma_fn = sigma_fn
        self.amplitude_params = amplitude_params
        self.sigma_params = sigma_params
        if kernel_trainable:
            self.all_parameters = [amplitude_params, sigma_params, kernel]
        else:
            self.all_parameters = [amplitude_params, sigma_params]
        
    def normalization_factor(self, xv, yv, a):
        dx = xv[0,1] - xv[0,0]
        dy = yv[1,0] - yv[0,0]
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        return self.amplitude_fn(a, self.amplitude_params) * self.dx0 * self.dy0 * self.sigma_fn(a, self.sigma_params)**2 * torch.abs(self.kernel).sum()  / dx / dy
        
    def _meshgrid_to_1Dtensor(self, xv, yv, a):
        lx, ly = xv.shape
        xv = xv.unsqueeze(0).repeat(a.shape[0],1,1).ravel()
        yv = yv.unsqueeze(0).repeat(a.shape[0],1,1).ravel()
        a = a.repeat(1,lx,ly).ravel()
        return xv, yv, a
        
    def _interpolate_kernel(self, xv, yv, a):
        num_psfs = len(a)
        lx, ly = xv.shape
        xv, yv, a = self._meshgrid_to_1Dtensor(xv, yv, a)
        sigma_x = self.dx0 * self.sigma_fn(a, self.sigma_params)
        sigma_y = self.dy0 * self.sigma_fn(a, self.sigma_params)
        return grid_sample(
            torch.abs(self.kernel).unsqueeze(0).unsqueeze(0),
            torch.stack([2*xv/(self.kernel.shape[0]*sigma_x),
                         2*yv/(self.kernel.shape[1]*sigma_y)],
                        dim=-1).unsqueeze(0).unsqueeze(0),
            mode = 'bicubic',
            align_corners = False
        ).squeeze().reshape(num_psfs,lx,ly)
    
    def __call__(self, xv, yv, a, normalize = False):
        N = self.normalization_factor(xv, yv, a) if normalize else 1
        a = torch.clamp(a, self.a_min, self.a_max)
        a = a.reshape(-1,1,1)
        return self.amplitude_fn(a, self.amplitude_params) * self._interpolate_kernel(xv, yv, a) / N

class NGonKernel2D(Kernel2D):
    def __init__(
        self,
        N_sides: int,
        Nx: int,
        collimator_width: float,
        amplitude_fn,
        sigma_fn,
        amplitude_params,
        sigma_params,
        a_min = -torch.inf,
        a_max = torch.inf,
        rot=0
        
    ):
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
        
    def normalization_constant(self, xv, yv, a):
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
    
    def __call__(self, xv, yv, a, normalize=False):
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
            
            
            
