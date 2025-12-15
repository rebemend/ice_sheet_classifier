#!/usr/bin/env python3
"""
Create synthetic test data for ice shelf classifier testing.
"""

import numpy as np
from scipy.io import savemat
import os

# Set random seed for reproducibility
np.random.seed(42)

# Grid parameters
nx, ny = 50, 40  # Small grid for testing
x_range = [0, 50000]  # 50 km
y_range = [0, 40000]  # 40 km

# Create coordinate grids
x = np.linspace(x_range[0], x_range[1], nx)
y = np.linspace(y_range[0], y_range[1], ny)
X, Y = np.meshgrid(x, y)

print(f"Creating synthetic data on {ny}x{nx} grid")

# Create synthetic velocity field
# Higher velocities toward the right (calving front)
u_base = 50 + 200 * (X - x_range[0]) / (x_range[1] - x_range[0])  # 50-250 m/year
v_base = 10 * np.sin(2 * np.pi * Y / (y_range[1] - y_range[0]))   # Some transverse flow

# Add some noise
u = u_base + np.random.normal(0, 10, u_base.shape)
v = v_base + np.random.normal(0, 5, v_base.shape)

# Ice thickness (decreases toward calving front)
h_base = 800 - 300 * (X - x_range[0]) / (x_range[1] - x_range[0])  # 500-800 m
h = h_base + np.random.normal(0, 20, h_base.shape)
h = np.maximum(h, 100)  # Ensure positive thickness

# Compute strain rates (simple finite differences)
dx = x[1] - x[0]
dy = y[1] - y[0]

dudx = np.gradient(u, dx, axis=1)
dudy = np.gradient(u, dy, axis=0) 
dvdx = np.gradient(v, dx, axis=1)
dvdy = np.gradient(v, dy, axis=0)

# Velocity magnitude
speed = np.sqrt(u**2 + v**2)

print(f"Velocity range: {np.min(speed):.1f} - {np.max(speed):.1f} m/year")
print(f"Strain rate range: {np.min(dudx):.2e} - {np.max(dudx):.2e} /year")

# Save DIFFICE-style data
os.makedirs('test_data/diffice_amery', exist_ok=True)

diffice_data = {
    'x': X,
    'y': Y, 
    'u': u,
    'v': v,
    'h': h,
    'dudx': dudx,
    'dvdy': dvdy,
    'dudy': dudy,
    'dvdx': dvdx,
    'speed': speed
}

np.savez_compressed('test_data/diffice_amery/amery_data.npz', **diffice_data)
print("DIFFICE test data saved to: test_data/diffice_amery/amery_data.npz")

# Create synthetic viscosity data
# Horizontal viscosity (varies with position and strain rate)
mu_base = 1e14 * (1 + 2 * np.abs(dudx) / 1e-4)  # Strain-dependent viscosity
mu = mu_base * (1 + 0.2 * np.random.random(mu_base.shape))

# Vertical viscosity (generally lower, creating anisotropy)
eta_base = mu / (2 + X / x_range[1])  # Anisotropy increases toward calving front
eta = eta_base * (1 + 0.15 * np.random.random(eta_base.shape))

print(f"Horizontal viscosity range: {np.min(mu):.2e} - {np.max(mu):.2e} Pa⋅s")
print(f"Vertical viscosity range: {np.min(eta):.2e} - {np.max(eta):.2e} Pa⋅s")
print(f"Anisotropy ratio range: {np.min(mu/eta):.1f} - {np.max(mu/eta):.1f}")

# Save MATLAB-style viscosity data
viscosity_data = {
    'mu': mu,
    'eta': eta,
    'x': X,
    'y': Y
}

savemat('test_data/results.mat', viscosity_data)
print("Viscosity test data saved to: test_data/results.mat")

print("\nTest data creation complete!")
print("You can now run the scripts with:")
print("  --diffice_data test_data/diffice_amery")
print("  --viscosity_data test_data/results.mat")