import pytest
import torch
from spectpsftoolbox.kernel1d import FunctionKernel1D
from spectpsftoolbox.kernel2d import NGonKernel2D, FunctionalKernel2D
from spectpsftoolbox.utils import get_kernel_meshgrid
from spectpsftoolbox.operator2d import Kernel2DOperator, GaussianOperator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_1Dkernel():
    amplitude_fn = lambda a, bs: bs[0]*torch.exp(-bs[1]*a)
    sigma_fn = lambda a, bs: bs[0]*(a+0.1)
    amplitude_params = torch.tensor([2,0.1], device=device, dtype=torch.float32)
    sigma_params = torch.tensor([0.3], device=device, dtype=torch.float32)
    kernel_fn = lambda x: torch.exp(-torch.abs(x))
    kernel1D = FunctionKernel1D(kernel_fn, amplitude_fn, sigma_fn, amplitude_params, sigma_params)
    x = torch.linspace(-5,5,100).to(device)
    a = torch.linspace(1,10,5).to(device)
    kernel_value = kernel1D(x, a)
    
def test_FunctionalKernel2D():
    Nx0 = 255
    dx0 = 0.05
    x = y = torch.arange(-(Nx0-1)/2, (Nx0+1)/2, 1).to(device) * dx0
    xv, yv = torch.meshgrid(x, y, indexing='xy')
    kernel_fn = lambda xv, yv: torch.exp(-torch.abs(xv))*torch.exp(-torch.abs(yv)) * torch.sin(xv*3)**2 * torch.cos(yv*3)**2
    amplitude_fn = lambda a, bs: bs[0]*torch.exp(-bs[1]*a)
    sigma_fn = lambda a, bs: bs[0]*(a+0.1)
    amplitude_params = torch.tensor([2,0.1], device=device, dtype=torch.float32)
    sigma_params = torch.tensor([0.3], device=device, dtype=torch.float32)
    # Define the kernel
    kernel2D = FunctionalKernel2D(kernel_fn, amplitude_fn, sigma_fn, amplitude_params, sigma_params)
    a = torch.linspace(1,10,5).to(device)
    kernel = kernel2D(xv, yv, a, normalize=True)
    
def test_NGonKernel2D():
    collimator_length = 2.405 
    collimator_width = 0.254 #flat side to flat side
    sigma_fn = lambda a, bs: (bs[0]+a) / bs[0] 
    sigma_params = torch.tensor([collimator_length], requires_grad=True, dtype=torch.float32, device=device)
    # Set amplitude to 1
    amplitude_fn = lambda a, bs: torch.ones_like(a)
    amplitude_params = torch.tensor([1.], requires_grad=True, dtype=torch.float32, device=device)
    ngon_kernel = NGonKernel2D(
        N_sides = 6, # sides of polygon
        Nx = 255, # resolution of polygon
        collimator_width=collimator_width, # width of polygon
        amplitude_fn=amplitude_fn,
        sigma_fn=sigma_fn,
        amplitude_params=amplitude_params,
        sigma_params=sigma_params,
        rot=90
    )
    Nx0 = 255
    dx0 = 0.048
    x = y = torch.arange(-(Nx0-1)/2, (Nx0+1)/2, 1).to(device) * dx0
    xv, yv = torch.meshgrid(x, y, indexing='xy')
    distances = torch.tensor([1,5,10,15,20,25], dtype=torch.float32, device=device)
    kernel = ngon_kernel(xv, yv, distances, normalize=True).cpu().detach()
    
def test_Operator1():
    # Tests Kernel2DOperator, GaussianOperator, and Operator __mult__
    # -------------------
    # Collimator Component
    # -------------------
    collimator_length = 2.405 
    collimator_width = 0.254 #flat side to flat side
    mu = 28.340267562430935
    sigma_fn = lambda a, bs: (bs[0]+a) / bs[0] 
    sigma_params = torch.tensor([collimator_length-2/mu], requires_grad=True, dtype=torch.float32, device=device)
    # Set amplitude to 1
    amplitude_fn = lambda a, bs: torch.ones_like(a)
    amplitude_params = torch.tensor([1.], requires_grad=True, dtype=torch.float32, device=device)
    ngon_kernel = NGonKernel2D(
        N_sides = 6, # sides of polygon
        Nx = 255, # resolution of polygon
        collimator_width=collimator_width, # width of polygon
        amplitude_fn=amplitude_fn,
        sigma_fn=sigma_fn,
        amplitude_params=amplitude_params,
        sigma_params=sigma_params,
        rot=90
    )
    ngon_operator = Kernel2DOperator(ngon_kernel)
    # -------------------
    # Detector component
    # -------------------
    intrinsic_sigma = 0.1614 # typical for NaI 140keV detection
    gauss_amplitude_fn = lambda a, bs: torch.ones_like(a)
    gauss_sigma_fn = lambda a, bs: bs[0]*torch.ones_like(a)
    gauss_amplitude_params = torch.tensor([1.], requires_grad=True, dtype=torch.float32, device=device)
    gauss_sigma_params = torch.tensor([intrinsic_sigma], requires_grad=True, device=device, dtype=torch.float32)
    scint_operator = GaussianOperator(
        gauss_amplitude_fn,
        gauss_sigma_fn,
        gauss_amplitude_params,
        gauss_sigma_params,
    )
    # Total combined:
    psf_operator = scint_operator * ngon_operator
    Nx0 = 512
    dx0 = 0.24
    x = y = torch.arange(-(Nx0-1)/2, (Nx0+1)/2, 1).to(device) * dx0
    xv, yv = torch.meshgrid(x, y, indexing='xy')
    distances = torch.arange(0.36, 57.9600, 0.48).to(device)
    # Get kernel meshgrid
    k_width = 24 #cm
    xv_k, yv_k = get_kernel_meshgrid(xv, yv, k_width)
    # Create input with point source at origin
    input = torch.zeros_like(xv).unsqueeze(0).repeat(distances.shape[0], 1, 1)
    input[:,256,256] = 1
    output = psf_operator(input, xv_k, yv_k, distances, normalize=True)