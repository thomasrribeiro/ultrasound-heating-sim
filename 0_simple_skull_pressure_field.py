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
from matplotlib.patches import Rectangle

# %% Simulation Parameters
# Transducer parameters
freq = 2e6  # 2 MHz frequency
num_cycles = 3  # 3 cycle pulse
pitch = 208e-6  # 208 µm pitch
num_elements_x = 140  # columns
num_elements_y = 64  # rows

# Grid parameters
dx = pitch  # spatial step [m]
dy = dx
dz = dx

# Grid size - adjusted to use numbers that factor into small primes (2, 3, 5)
pml_size = 10
Nx = 256 - pml_size * 2
Ny = 128 - pml_size * 2
Nz = 128 - pml_size * 2

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

# Visualize the grid structure
plt.figure(figsize=(12, 8))
plt.imshow(medium.sound_speed[:, Ny // 2, :].T, aspect="auto", cmap="viridis")
plt.colorbar(label="Sound Speed [m/s]")
plt.title("Sound Speed Distribution in Tissue Layers\n(XZ plane at Y = Ny/2)")
plt.xlabel("X position [grid points]")
plt.ylabel("Z position [grid points]")

# Calculate source dimensions
source_x_size = num_elements_x * (pitch / dx)  # Convert transducer size to grid points
source_y_size = num_elements_y * (pitch / dy)

# Add annotations for layers
plt.axhline(y=z_start, color="w", linestyle="--", alpha=0.5)
plt.axhline(y=z_start + skin_points, color="w", linestyle="--", alpha=0.5)
plt.axhline(
    y=z_start + skin_points + skull_points, color="w", linestyle="--", alpha=0.5
)

# Add layer labels
plt.text(Nx // 2, z_start - 5, "Water", color="w", ha="center", fontsize=12)
plt.text(
    Nx // 2, z_start + skin_points // 2, "Skin", color="w", ha="center", fontsize=12
)
plt.text(
    Nx // 2,
    z_start + skin_points + skull_points // 2,
    "Skull",
    color="w",
    ha="center",
    fontsize=12,
)
plt.text(
    Nx // 2,
    z_start + skin_points + skull_points + 10,
    "Brain",
    color="w",
    ha="center",
    fontsize=12,
)

# Add transducer array visualization
x_start = round((Nx - source_x_size) / 2)
transducer_height = 3  # Height of the transducer visualization in grid points
rect = Rectangle(
    (x_start, 10),
    source_x_size,
    transducer_height,
    facecolor="red",
    alpha=0.5,
    edgecolor="white",
)
plt.gca().add_patch(rect)
plt.text(Nx // 2, 8, "Transducer Array", color="red", ha="center", fontsize=12)

plt.show()

# %% Setup Source
# Create time varying source
source_mag = 1e6  # [Pa]
source_signal = tone_burst(1 / kgrid.dt, freq, num_cycles)
source_signal = source_mag * source_signal

# Define source mask for plane wave
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
sensor = kSensor(sensor_mask, record=["p"])

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

# %%
sensor_data["p"].shape
Nt = sensor_data["p"].shape[0]
d = sensor_data["p"].reshape(Nt, Nx, Ny, Nz - z_start, order="F")

# %%
plt.figure(figsize=(10, 8))
plt.imshow(d[190, :, Ny // 2, :].T, cmap="coolwarm")
plt.colorbar(label="Pressure [Pa]")
plt.title("Pressure Field in Tissue Layers")
plt.xlabel("X position [grid points]")
plt.ylabel("Z position [grid points]")
plt.show()

# %%
# Save the simulation data to a numpy file
np.savez(
    "simulation_data.npz",
    pressure_data=sensor_data["p"],
    grid_params={
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "z_start": z_start,
    },
    time_params={"dt": dt, "t_end": t_end, "Nt": Nt},
)
print("Simulation data saved to simulation_data.npz")


# # # %%
# # d = sensor_data["p_max"].reshape(Nx, Nz - z_start, order="F")

# # plt.figure(figsize=(10, 8))
# # plt.imshow(d.T, cmap="coolwarm")
# # plt.colorbar(label="Pressure [Pa]")
# # plt.title("Pressure Field in Tissue Layers")
# # plt.xlabel("X position [grid points]")
# # plt.ylabel("Z position [grid points]")
# # plt.show()

# # %%
# sensor_data["p"].shape
# nt = sensor_data["p"].shape[0]

# d = sensor_data["p"].reshape(nt, Nx, Nz - z_start, order="F")

# # %%
# plt.figure(figsize=(10, 8))
# plt.imshow(d[90].T, cmap="coolwarm")
# plt.colorbar(label="Pressure [Pa]")
# plt.title("Pressure Field in Tissue Layers")
# plt.xlabel("X position [grid points]")
# plt.ylabel("Z position [grid points]")
# plt.show()

# # %%
# import matplotlib.animation as animation

# # Get global min/max for consistent colorbar
# vmin = d.min()
# vmax = d.max()

# # Create animation
# fig, ax = plt.subplots(figsize=(10, 8))
# im = ax.imshow(d[0].T, cmap="coolwarm", vmin=vmin, vmax=vmax)
# plt.colorbar(im, label="Pressure [Pa]")
# ax.set_xlabel("X position [grid points]")
# ax.set_ylabel("Z position [grid points]")


# def update(frame):
#     im.set_array(d[4 * frame].T)
#     ax.set_title(f"Pressure Field in Tissue Layers - Time step {2 * frame}")
#     return [im]


# anim = animation.FuncAnimation(fig, update, frames=nt // 4, interval=20, blit=True)
# anim.save("pressure_wave.mp4", writer="ffmpeg")
# plt.close()

# print("Video saved as pressure_wave.mp4")

# # %%
