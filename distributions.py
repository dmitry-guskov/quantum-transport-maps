# File: src/datasets.py

import torch
import numpy as np

import torch

def gauss_to_gauss(num_samples=1000, mean_x=None, mean_y=None, dim=2, cov_scale=0.01):
    """
    Generates two datasets (x and y) where each is sampled from a single Gaussian distribution.
    
    x is sampled from N(mean_x, cov_scale * I) and y is sampled from N(mean_y, cov_scale * I).

    Args:
        num_samples (int): Number of samples for each dataset.
        mean_x (tuple, list, or torch.Tensor): Mean for the x dataset, defaults to [0.5, 0.0, ...] (dim-dimensional).
        mean_y (tuple, list, or torch.Tensor): Mean for the y dataset, defaults to [0.0, -0.5, ...] (dim-dimensional).
        dim (int): Dimensionality of the Gaussian distributions.
        cov_scale (float): Scale factor for the covariance matrix.

    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, dim)
    """
    # Default means if not provided
    if mean_x is None:
        mean_x = torch.tensor([0.5] + [0.0] * (dim - 1))
    if mean_y is None:
        mean_y = torch.tensor([0.0] + [-0.5] + [0.0] * (dim - 2))

    # Convert lists/tuples to tensors if necessary
    mean_x = torch.tensor(mean_x, dtype=torch.float32)
    mean_y = torch.tensor(mean_y, dtype=torch.float32)

    # Ensure dimensionality consistency
    assert mean_x.shape[0] == dim, f"mean_x should have {dim} dimensions, got {mean_x.shape[0]}"
    assert mean_y.shape[0] == dim, f"mean_y should have {dim} dimensions, got {mean_y.shape[0]}"

    # Isotropic covariance matrix (cov_scale * I)
    cov = torch.eye(dim) * cov_scale
    
    x_data = torch.distributions.MultivariateNormal(mean_x, cov).sample((num_samples,))
    y_data = torch.distributions.MultivariateNormal(mean_y, cov).sample((num_samples,))
    
    return x_data, y_data


def two_gaussians_to_two_gaussians(num_samples=1000, mean_x=None, mean_y=None):
    """
    Generates two datasets (x and y) where each is a mixture of two Gaussian distributions.
    
    For x:
      - 50% samples from N([0.5, 0.0], 0.01*I)
      - 50% samples from N([-0.5, 0.5], 0.01*I)
    
    For y:
      - 50% samples from N([0.0, -0.5], 0.01*I)
      - 50% samples from N([0.5, 0.5], 0.01*I)

    Args:
        num_samples (int): Total number of samples for each dataset.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    num_samples_half = num_samples // 2

    # Mixture for x
    if mean_x is None:
        mean_x1 = torch.tensor([0.5, 0.0])
        mean_x2 = torch.tensor([-0.5, 0.5])
    else:
        mean_x1 = mean_x[0]
        mean_x2 = mean_x[1]
        
    if mean_y is None:
        mean_y1 = torch.tensor([0.0, -0.5])
        mean_y2 = torch.tensor([0.5, 0.5])
    else:
        mean_y1 = mean_y[0]
        mean_y2 = mean_y[1]
        
    
    cov_x = torch.tensor([[0.01, 0.0], [0.0, 0.01]])
    
    x_data1 = torch.distributions.MultivariateNormal(mean_x1, cov_x).sample((num_samples_half,))
    x_data2 = torch.distributions.MultivariateNormal(mean_x2, cov_x).sample((num_samples - num_samples_half,))
    x_data = torch.cat([x_data1, x_data2], dim=0)

    # Mixture for y

    cov_y = torch.tensor([[0.01, 0.0], [0.0, 0.01]])
    
    y_data1 = torch.distributions.MultivariateNormal(mean_y1, cov_y).sample((num_samples_half,))
    y_data2 = torch.distributions.MultivariateNormal(mean_y2, cov_y).sample((num_samples - num_samples_half,))
    y_data = torch.cat([y_data1, y_data2], dim=0)

    return x_data, y_data

def linear_relation(num_samples=1000, b=0.5):
    """
    Generates a dataset x sampled from a Gaussian distribution and computes y = x + b.
    The bias b is added to each dimension of x.

    Args:
        num_samples (int): Number of samples.
        b (float): Bias added to x to form y.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[0.01, 0.0], [0.0, 0.01]])
    
    x_data = torch.distributions.MultivariateNormal(mean, cov).sample((num_samples,))
    y_data = x_data + b
    return x_data, y_data

def circle_to_circle(num_samples=1000, noise=0.01, radius_x=1.0, radius_y=0.5, 
                     center_x=(0.0, 0.0), center_y=(0.0, 0.0)):
    """
    Generates two datasets (x and y) where each is sampled from a circle distribution with added noise.
    
    x_data is generated from points on a circle with radius `radius_x` centered at `center_x`.
    y_data is generated from points on a circle with radius `radius_y` centered at `center_y`.

    Args:
        num_samples (int): Number of samples for each dataset.
        noise (float): Standard deviation of Gaussian noise added to the points.
        radius_x (float): Radius of the circle for x_data.
        radius_y (float): Radius of the circle for y_data.
        center_x (tuple): Center coordinates for the x circle.
        center_y (tuple): Center coordinates for the y circle.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    angles_x = torch.rand(num_samples) * 2 * torch.pi
    angles_y = torch.rand(num_samples) * 2 * torch.pi
    x_data = torch.stack([radius_x * torch.cos(angles_x), radius_x * torch.sin(angles_x)], dim=1)
    y_data = torch.stack([radius_y * torch.cos(angles_y), radius_y * torch.sin(angles_y)], dim=1)
    
    # Shift the circles to their centers
    x_data = x_data + torch.tensor(center_x, dtype=x_data.dtype)
    y_data = y_data + torch.tensor(center_y, dtype=y_data.dtype)
    
    # Add noise
    x_data = x_data + torch.randn_like(x_data) * noise
    y_data = y_data + torch.randn_like(y_data) * noise
    
    return x_data, y_data

def moons_to_moons(num_samples=1000, noise=0.05, shift=(0.5, 0.0)):
    """
    Generates two datasets (x and y) using a two-moons structure.
    
    x_data is generated as a two-moons dataset.
    y_data is generated by applying a constant shift to x_data.

    Args:
        num_samples (int): Total number of samples for the combined moons.
        noise (float): Standard deviation of Gaussian noise added to the points.
        shift (tuple): Shift applied to x_data to generate y_data.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    n = num_samples // 2
    
    # Generate first moon: angles in [0, π]
    theta1 = torch.rand(n) * torch.pi
    moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
    
    # Generate second moon: angles in [0, π] (flipped and shifted)
    theta2 = torch.rand(n) * torch.pi
    moon2 = torch.stack([1 - torch.cos(theta2), -torch.sin(theta2) - 0.5], dim=1)
    
    x_data = torch.cat([moon1, moon2], dim=0)
    x_data = x_data + torch.randn_like(x_data) * noise
    
    shift_tensor = torch.tensor(shift, dtype=x_data.dtype)
    y_data = x_data + shift_tensor
    
    return x_data, y_data

def nonlinear_warp(num_samples=1000, noise=0.01, warp_strength=1.0):
    """
    Generates a dataset x sampled from a Gaussian distribution and computes y as a nonlinear
    warp of x. The warp is applied by modulating the first coordinate with a sine function of
    the second coordinate.
    
    Specifically:
        y_data[:, 0] = x_data[:, 0] + warp_strength * sin(x_data[:, 1])
        y_data[:, 1] = x_data[:, 1]

    Args:
        num_samples (int): Number of samples.
        noise (float): Standard deviation of Gaussian noise added to x_data.
        warp_strength (float): Strength of the nonlinearity applied to x_data.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[0.01, 0.0], [0.0, 0.01]])
    
    x_data = torch.distributions.MultivariateNormal(mean, cov).sample((num_samples,))
    x_data = x_data + torch.randn_like(x_data) * noise
    
    y_data = x_data.clone()
    y_data[:, 0] = x_data[:, 0] + warp_strength * torch.sin(x_data[:, 1])
    
    return x_data, y_data

def spiral_to_spiral(num_samples=1000, noise=0.05, rotations=2, scale=0.1, shift=(0.5, 0.5)):
    """
    Generates two datasets (x and y) where each is sampled from a spiral pattern.
    
    x_data is generated along a spiral curve with the specified number of rotations and scale.
    y_data is produced by applying a constant shift to x_data.
    
    Args:
        num_samples (int): Number of samples for each dataset.
        noise (float): Standard deviation of Gaussian noise added to the points.
        rotations (float): Number of rotations in the spiral.
        scale (float): Scale factor controlling the spiral's tightness.
        shift (tuple): Shift applied to x_data to generate y_data.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    t = torch.linspace(0, 2 * torch.pi * rotations, num_samples)
    x_spiral = scale * t * torch.cos(t)
    y_spiral = scale * t * torch.sin(t)
    x_data = torch.stack([x_spiral, y_spiral], dim=1)
    x_data = x_data + torch.randn_like(x_data) * noise
    
    shift_tensor = torch.tensor(shift, dtype=x_data.dtype)
    y_data = x_data + shift_tensor
    
    return x_data, y_data

def uniform_to_spiral(num_samples=1000, noise=0.01, scale=1, rotations =2):
    """
    Generates two datasets (x and y) where each is sampled from a spiral pattern.
    
    x_data is generated along a spiral curve with the specified number of rotations and scale.
    y_data is produced by applying a constant shift to x_data.
    
    Args:
        num_samples (int): Number of samples for each dataset.
        noise (float): Standard deviation of Gaussian noise added to the points.
        rotations (float): Number of rotations in the spiral.
        scale (float): Scale factor controlling the spiral's tightness.
        shift (tuple): Shift applied to x_data to generate y_data.
    
    Returns:
        tuple: (x_data, y_data) each of shape (num_samples, 2)
    """
    
    x_data = torch.stack([torch.rand(num_samples)-0.5,torch.rand(num_samples)-0.5], dim=1)
    
    t = torch.sum(x_data+0.5,dim=1)*torch.pi*rotations

    adjust = 1/(4*torch.pi*rotations)
    
    x_spiral = scale * t * torch.cos(t)*adjust
    y_spiral = scale * t * torch.sin(t)*adjust
    y_data = torch.stack([x_spiral, y_spiral], dim=1)
    y_data = y_data + torch.randn_like(y_data) * noise
    
    return x_data, y_data