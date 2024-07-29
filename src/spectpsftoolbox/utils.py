import torch

def get_kernel_meshgrid(xv_input, yv_input, k_width):
    dx = xv_input[0,1] - xv_input[0,0]
    dy = yv_input[1,0] - yv_input[0,0]
    x_kernel = torch.arange(0,k_width/2,dx).to(xv_input.device)
    x_kernel = torch.cat([-x_kernel.flip(dims=(0,))[:-1], x_kernel])
    y_kernel = torch.arange(0,k_width/2,dy).to(xv_input.device)
    y_kernel = torch.cat([-y_kernel.flip(dims=(0,))[:-1], y_kernel])
    xv_kernel, yv_kernel = torch.meshgrid(x_kernel, y_kernel, indexing='xy')
    return xv_kernel, yv_kernel