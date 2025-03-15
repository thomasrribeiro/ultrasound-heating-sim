# %% Imports
import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

# %% Simulation Parameters
# Transducer parameters
freq = 2e6  # 2 MHz frequency
num_cycles = 3  # 3 cycle pulse
pitch = 208e-6  # 208 µm pitch
num_elements_x = 140  # columns
num_elements_y = 64  # rows

# Grid parameters
dx = pitch / 2  # spatial step [m]
dy = dx
dz = dx

# Grid size - adjusted to use numbers that factor into small primes (2, 3, 5)
pml_size = 10
Nx = 256 - pml_size * 2
Ny = 256 - pml_size * 2
Nz = 256 - pml_size * 2

# %% Create k-Wave Grid
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

# %% Define Medium Properties
# Layer 1: Skin
skin_sound_speed = 1610  # [m/s]
skin_density = 1090  # [kg/m^3]
skin_thickness = 2e-3  # 2 mm

# Layer 2: Skull
skull_sound_speed = 3200  # [m/s]
skull_density = 1900  # [kg/m^3]
skull_thickness = 7e-3  # 7 mm

# Layer 3: Brain
brain_sound_speed = 1560  # [m/s]
brain_density = 1040  # [kg/m^3]

# Calculate the time step using a smaller CFL number for better stability
cfl = 0.2  # reduced from 0.3 for better stability
c_max = skull_sound_speed  # use maximum sound speed in the medium for stability
dt = cfl * dx / c_max
t_end = 25e-6  # [s] slightly increased to account for longer propagation time
kgrid.makeTime(c_max, t_end=t_end)

# Create the medium properties
medium = kWaveMedium(
    sound_speed=brain_sound_speed * np.ones(kgrid.k.shape),
    density=brain_density * np.ones(kgrid.k.shape),
    alpha_coeff=0.75,
    alpha_power=1.5,
    BonA=6,
)

# %% Create Layered Medium
# Convert thicknesses to grid points
skin_points = round(skin_thickness / dx)
skull_points = round(skull_thickness / dx)

# Create the layers
z_start = 20  # Start position of first layer
medium.sound_speed[:, :, z_start : z_start + skin_points] = skin_sound_speed
medium.density[:, :, z_start : z_start + skin_points] = skin_density

medium.sound_speed[
    :, :, z_start + skin_points : z_start + skin_points + skull_points
] = skull_sound_speed
medium.density[:, :, z_start + skin_points : z_start + skin_points + skull_points] = (
    skull_density
)

# %% Setup Source
# Create time varying source
source_mag = 1e6  # [Pa]
source_signal = tone_burst(1 / kgrid.dt, freq, num_cycles)
source_signal = source_mag * source_signal

# Define source mask for plane wave
source_x_size = num_elements_x * (pitch / dx)  # Convert transducer size to grid points
source_y_size = num_elements_y * (pitch / dy)

source_mask = np.zeros(kgrid.k.shape)
x_start = round((Nx - source_x_size) / 2)
y_start = round((Ny - source_y_size) / 2)
source_mask[
    x_start : x_start + int(source_x_size), y_start : y_start + int(source_y_size), 10
] = 1

# Create source structure
source = kSource()
source.p_mask = source_mask
source.p = source_signal

# %% Setup Sensor
# Create sensor mask
sensor_mask = np.zeros(kgrid.k.shape)
sensor_mask[:, :, z_start:] = 1  # Record pressure in tissue layers

# Create sensor object
sensor = kSensor(sensor_mask)

# %% Run Simulation
# Set input options
simulation_options = SimulationOptions(
    pml_inside=False,
    pml_size=pml_size,
    data_cast="single",
    save_to_disk=True,
    data_recast=True,
    save_to_disk_exit=False,
)

# Run the simulation
sensor_data = kspaceFirstOrder3D(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=SimulationExecutionOptions(is_gpu_simulation=True),
)

# %% Visualize Results
plt.figure(figsize=(10, 8))
plt.imshow(np.squeeze(sensor_data["p"][:, Ny // 2, :]).T, aspect="auto")
plt.colorbar(label="Pressure [Pa]")
plt.title("Pressure Field (XZ plane at Y = Ny/2)")
plt.xlabel("X position [grid points]")
plt.ylabel("Z position [grid points]")
plt.show()
