import torch
import numpy as np
from torch.nn.functional import pad
from torch.nn.functional import conv1d, conv2d
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from torch.nn.functional import grid_sample
from fft_conv_pytorch import fft_conv
from .kernel1d import GaussianKernel1D
import dill

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2)) 

def compute_pad_size_padded(width: int):
    a = (np.sqrt(2) - 1)/2
    if width%2==0:
        width_old = int(2*np.floor((width/2)/(1+2*a)))
    else:
        width_old = int(2*np.floor(((width-1)/2)/(1+2*a)))
    return int((width-width_old)/2)

def pad_object(object: torch.Tensor, mode='constant'):
    pad_size = compute_pad_size(object.shape[-2]) 
    return pad(object, [pad_size,pad_size,pad_size,pad_size], mode=mode)

def unpad_object(object: torch.Tensor):
    pad_size = compute_pad_size_padded(object.shape[-2])
    return object[:,pad_size:-pad_size,pad_size:-pad_size]

class Operator:
    def normalization_constant(self, xv, yv, a):
        ...
    def __call__(self, input, xv, yv, a):
        return input * self._area(xv,yv)
    def _area(self, xv, yv):
        return (xv[0,1]-xv[0,0])*(yv[1,0]-yv[0,0])
    def set_device(self, device):
        for p in self.params:
            p.data = p.data.to(device)
    def detach(self):
        for p in self.params:
            p.detach_()
    def set_requires_grad(self):
        for p in self.params:
            p.requires_grad_(True)
    def save(self, path):
        self.set_device('cpu')
        self.detach()
        dill.dump(self, open(path, 'wb'))
        
    def normalize(self, input, xv, yv, a):
        return input / self.normalization_constant(xv, yv, a)
        
    def __add__(self, other):
        def combined_operator(x, *args, **kwargs):
            return self(x, *args, **kwargs) + other(x, *args, **kwargs)
        return CombinedOperator(combined_operator, [self, other], type='additive')
    
    def __mul__(self, other):
        def combined_operator(x, *args, **kwargs):
            return self(other(x, *args, **kwargs), *args, **kwargs)
        return CombinedOperator(combined_operator, [self,other], type='sequential')
    
class CombinedOperator(Operator):
    def __init__(self, func, operators, type):
        self.params = [*operators[0].params, *operators[1].params]
        self.func = func
        self.type = type
        self.operators = operators
        
    def set_device(self, device):
        for operator in self.operators:
            operator.set_device(device)
            
    def detach(self):
        for operator in self.operators:
            operator.detach()
        
    def normalization_constant(self, xv, yv, a):
        if self.type=='additive':
            return self.operators[0].normalization_constant(xv, yv, a) + self.operators[1].normalization_constant(xv, yv, a)
        else:
            return self.operators[0].normalization_constant(xv, yv, a) * self.operators[1].normalization_constant(xv, yv, a)

    def __call__(self, input, xv, yv, a, normalize=False):
        if normalize:
            return self.func(input, xv, yv, a) / self.normalization_constant(xv, yv, a)
        else:
            return self.func(input,xv, yv, a)
    
class Rotate1DConvOperator(Operator):
    def __init__(
        self,
        kernel1D,
        N_angles,
        additive = False,
        use_fft_conv = False,
        rot = 0
    ):
        self.params = kernel1D.params
        self.kernel1D = kernel1D
        self.N_angles = N_angles
        self.angles = [180*i/N_angles + rot for i in range(N_angles)]
        self.additive = additive
        self.angle_delta = 1e-4
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input):
        if self.use_fft_conv:
            return fft_conv(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        else:
            return conv1d(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(self, xv, yv, a):
        if self.additive:
            return (self.N_angles*self.kernel.sum(dim=-1)).unsqueeze(-1) * torch.sqrt(self._area(xv,yv))
        else:
            return ((self.kernel.sum(dim=-1))**self.N_angles).unsqueeze(-1) * self._area(xv,yv)
    
    def _rotate(self, input, angle):
        if abs(angle)<self.angle_delta:
            return input
        elif abs(angle%90)<self.angle_delta:
            return torch.rot90(input, int(angle//90), dims=[1,2])
        else:
            return rotate(input, angle, interpolation=InterpolationMode.BILINEAR)
            
    def _apply_additive(self, input):
        output = 0
        for angle in self.angles:
            output_i = self._rotate(input, angle)
            output_i = output_i.swapaxes(0,1)
            output_i = self._conv(output_i)
            output_i = output_i.swapaxes(0,1)
            output_i = self._rotate(output_i, -angle)
            output += output_i
        return output
    
    def _apply_regular(self, input):
        for angle in self.angles:
            input = self._rotate(input, angle)
            input = input.swapaxes(0,1)
            input = self._conv(input)
            input = input.swapaxes(0,1)
            input = self._rotate(input, angle)
        return input 
        
    def __call__(self, input, xv, yv, a, normalize=False):
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
    def __init__(
        self,
        kernel1D,
        N_angles,
        additive = False,
        use_fft_conv = False,
        rot = 0
    ):
        self.params = kernel1D.params
        self.kernel1D = kernel1D
        self.N_angles = N_angles
        self.angles = [90*i/N_angles + rot for i in range(N_angles)]
        self.additive = additive
        self.angle_delta = 1e-4
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input):
        if self.use_fft_conv:
            return fft_conv(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        else:
            return conv1d(input, self.kernel, padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(self, xv, yv, a):
        dx = xv[0,1] - xv[0,0]
        if self.additive:
            return (self.N_angles*self.kernel.sum(dim=-1)**2).unsqueeze(-1) * self._area(xv,yv)
        else:
            return ((self.kernel.sum(dim=-1)**2)**self.N_angles).unsqueeze(-1) * self._area(xv,yv)
    
    def _rotate(self, input, angle):
        if abs(angle)<self.angle_delta:
            return input
        elif abs(angle%90)<self.angle_delta:
            return torch.rot90(input, int(angle//90), dims=[1,2])
        else:
            return rotate(input, angle, interpolation=InterpolationMode.BILINEAR)
            
    def _apply_additive(self, input):
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
    
    def _apply_regular(self, input):
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
        
    def __call__(self, input, xv, yv, a, normalize=False):
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
    def __init__(
        self,
        kernel2D,
        use_fft_conv = False,
    ):
        self.params = kernel2D.params
        self.kernel2D = kernel2D
        self.use_fft_conv = use_fft_conv
        
    def _conv(self, input):
        if self.use_fft_conv:
            return fft_conv(input, self.kernel.unsqueeze(1), padding='same', groups=self.kernel.shape[0])
        else:
            return conv2d(input, self.kernel.unsqueeze(1), padding='same', groups=self.kernel.shape[0])
        
    def normalization_constant(self, xv, yv, a):
        return self.kernel2D.normalization_constant(xv, yv, a) * self._area(xv,yv)

    def __call__(self, input, xv, yv, a, normalize=False):
        self.kernel = self.kernel2D(xv,yv,a)
        if normalize:
            self.kernel = self.kernel / self.normalization_constant(xv, yv, a)
        return self._conv(input) * self._area(xv,yv)
    
    
class NearestKernelOperator(Operator):
    def __init__(
        self,
        psf_data,
        distances,
        dr0,
        use_fft_conv = True,
        grid_sample_mode = 'bilinear'
    ):
        self.psf_data = psf_data
        self.Nx0 = psf_data.shape[1]
        self.Ny0 = psf_data.shape[2]
        self.distances_original = distances
        self.dr0 = dr0
        self.use_fft_conv = use_fft_conv
        self.params = []
        self.grid_sample_mode = grid_sample_mode
        
    def set_device(self, device):
        # Override 
        self.psf_data = self.psf_data.to(device)
        self.distances_original = self.distances_original.to(device)
        
    def _conv(self, input, kernel):
        groups = input.shape[0]
        if self.use_fft_conv:
            return fft_conv(input.unsqueeze(0), kernel.unsqueeze(1), padding='same', groups=groups).squeeze()
        else:
            return conv2d(input.unsqueeze(0), kernel.unsqueeze(1), padding='same', groups=groups).squeeze()
    
    def _get_nearest_distance_idxs(self, distances):
        differences = torch.abs(distances[:, None] - self.distances_original)
        indices = torch.argmin(differences, dim=1)
        return self.psf_data[indices]
    
    def _get_kernel(self, xv, yv, a):
        dx = xv[0, 1] - xv[0, 0]
        psf = self._get_nearest_distance_idxs(a)
        grid = torch.stack([
            2*xv/(self.Nx0 * self.dr0[0]),
            2*yv/(self.Ny0 * self.dr0[1])],
            dim=-1).unsqueeze(0).repeat(a.shape[0], 1, 1, 1)
        return (dx/self.dr0[0])**2 * grid_sample(psf.unsqueeze(1), grid, align_corners=False, mode=self.grid_sample_mode).squeeze()
    
    def __call__(self, input, xv, yv, a, normalize=False):
        kernel = self._get_kernel(xv, yv, a)
        return self._conv(input, kernel).squeeze() 
    
    
class GaussianOperator(Operator):
    def __init__(
        self,
        amplitude_fn,
        sigma_fn,
        amplitude_params,
        sigma_params,
        a_min = -torch.inf,
        a_max = torch.inf,
        use_fft_conv = False,
    ):  
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
        
    def normalization_constant(self, xv, yv, a):
        return self.kernel1D.normalization_constant(xv[0], a).unsqueeze(1)**2 * self._area(xv,yv)
        
    def _conv(self, input, kernel1D):
        if self.use_fft_conv:
            return fft_conv(input, kernel1D, padding='same', groups=kernel1D.shape[0])
        else:
            return conv1d(input, kernel1D, padding='same', groups=kernel1D.shape[0])
        
    def __call__(self, input, xv, yv, a, normalize=False):
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