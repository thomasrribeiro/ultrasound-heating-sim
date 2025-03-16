# %% Imports
try:
    %load_ext autoreload
    %autoreload 2
except:
    pass
from simulation_config import SimulationConfig
from skull_pressure_simulator import SkullPressureSimulator
from visualization import (
    plot_medium_properties,
    plot_intensity_field,
)
import matplotlib.pyplot as plt
import numpy as np

# %% Create simulation configuration
config = SimulationConfig()

# %% Initialize simulator
simulator = SkullPressureSimulator(config)

# %% Set up the grid
print("Setting up grid...")
simulator.setup_grid()

# %% Set up the medium
print("Setting up medium...")
simulator.setup_medium()

# %% Plot the medium properties
fig, ax = plot_medium_properties(simulator.medium.sound_speed, config)
plt.show()  # Interactive display in notebook
plt.close()

# %% Set up source and sensor
print("Setting up source and sensor...")
simulator.setup_source_sensor()

# %% Run the simulation
print("Running simulation...")
sensor_data = simulator.run_simulation(use_gpu=True)

# %% Process and reshape the pressure data
pressure_data = sensor_data["p"].reshape(
    -1,  # time steps
    config.acoustic.Nx,
    config.acoustic.Ny,
    config.acoustic.Nz - config.initial_tissue_z,
    order="F",
)

#%%
# plot max pressure
max_pressure = np.max(pressure_data, axis=0)
plt.imshow(1e-6 * max_pressure[:, config.acoustic.Ny // 2, :].T, cmap="coolwarm")
plt.colorbar(label="Max Pressure [MPa]")
plt.title("Max Pressure Field")
plt.xlabel("X position [grid points]")
plt.ylabel("Z position [grid points]")
plt.show()
plt.close()

# %% Compute intensity fields
print("Computing intensity fields...")
average_intensity = simulator.compute_intensity(pressure_data)


# %% Plot time-averaged intensity field
fig, ax = plot_intensity_field(
    average_intensity,
    config,
    title="Time-Averaged Intensity Field",
)
plt.show()
plt.close()

#%%
average_intensity.shape

# %% Save intensity data
np.save("data/average_intensity.npy", average_intensity)
