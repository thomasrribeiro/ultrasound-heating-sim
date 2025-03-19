# %% Imports and setup
"""
Main script for bioheat equation simulation using acoustic intensity data.
"""

try:
    import IPython

    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
except:
    pass

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

from config import SimulationConfig
from heat_simulator import BioheatSimulator
from heat_visualization import (
    plot_temperature_evolution,
    plot_temperature_field_slices,
    plot_tissue_properties,
    visualize_combined_results,
    make_temperature_video,
)

intensity_file = "data/average_intensity.npy"


# %% Define main simulation function
"""
Run a bioheat simulation using acoustic intensity data.

Args:
    intensity_file: Path to the .npy file containing acoustic intensity data
"""
# Check if intensity file exists
intensity_path = Path(intensity_file)
if not intensity_path.exists():
    raise FileNotFoundError(f"Intensity file not found: {intensity_file}")

# Create output directory
os.makedirs("data", exist_ok=True)

# %% Load data and initialize simulator
# Load the acoustic intensity data
print(f"Loading acoustic intensity data from {intensity_file}...")
intensity_data = np.load(intensity_file)

# Verify intensity data shape matches expected tissue domain
print(f"Intensity data shape: {intensity_data.shape}")
config = SimulationConfig()

# Initialize simulator
print("Initializing bioheat simulator...")
simulator = BioheatSimulator(config)

# %% Setup and visualize mesh
print("Setting up computational mesh...")
# This will now use only the tissue region for thermal simulation
simulator.setup_mesh()

# Visualize the layer map
print("Visualizing tissue layer map...")
layer_map = simulator.get_layer_map()
plt.figure(figsize=(10, 5))
plt.imshow(
    layer_map[:, layer_map.shape[1] // 2, :].cpu().numpy().T,
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="Tissue Type (0=skin, 1=skull, 2=brain)")
plt.title("Tissue Layer Map - Thermal Simulation (XZ mid-slice)")
plt.xlabel("X")
plt.ylabel("Z (tissue region only)")
plt.savefig("data/thermal_tissue_layer_map.png")
plt.close()

# %% Setup and visualize tissue properties
print("Setting up tissue properties...")
simulator.setup_tissue_properties()

# %% Run simulation
# Setup heat source using acoustic intensity
print("Setting up heat source from acoustic intensity...")
intensity_tensor = torch.tensor(intensity_data, device=simulator.device)
simulator.setup_heat_source(intensity_field=intensity_tensor)

# Run simulation
print("Running bioheat simulation...")
T_history, times, max_temps = simulator.run_simulation()

# %% Visualize results
print("Plotting results...")
# Temperature evolution
fig, _ = plot_temperature_evolution(times, max_temps)
plt.savefig("data/temperature_evolution.png")
plt.close(fig)

# Temperature distribution
fig, _ = plot_temperature_field_slices(T_history[-1], config)
plt.savefig("data/temperature_distribution.png")
plt.close(fig)

# Combined acoustic intensity and temperature visualization
fig, _ = visualize_combined_results(intensity_tensor, T_history[-1], config)
plt.savefig("data/acoustic_thermal_combined.png")
plt.close(fig)

# Save results
# print("Saving results to HDF5...")
# simulator.save_results(
#     "data/bioheat_results.h5", T_history[-1], times, max_temps, T_history
# )

# %%

# Create temperature evolution video using the full history
print("Creating temperature evolution video...")
make_temperature_video(
    T_history[::5], config, times[::5], "data/temperature_evolution.mp4"
)

print(f"Temperature history shape: {T_history.shape}")
print(f"Number of time points: {len(times)}")
print("Simulation complete!")

# %%
