import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple
from config import SimulationConfig
import matplotlib.animation as animation


def plot_medium_properties(
    medium_sound_speed: np.ndarray,
    config: SimulationConfig,
    slice_y: int | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the sound speed distribution in tissue layers."""
    if slice_y is None:
        slice_y = config.grid.Ny // 2

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(medium_sound_speed[:, slice_y, :].T, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Sound Speed [m/s]")
    ax.set_title("Sound Speed Distribution in Tissue Layers\n(XZ plane at Y = Ny/2)")
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    # Add layer annotations
    z_start = config.initial_tissue_z
    skin_points = int(config.skin.thickness / config.grid.dz)
    skull_points = int(config.skull.thickness / config.grid.dz)

    # Add horizontal lines for layer boundaries
    ax.axhline(y=z_start, color="w", linestyle="--", alpha=0.5)
    ax.axhline(y=z_start + skin_points, color="w", linestyle="--", alpha=0.5)
    ax.axhline(
        y=z_start + skin_points + skull_points, color="w", linestyle="--", alpha=0.5
    )

    # Add layer labels
    ax.text(
        config.grid.Nx // 2,
        z_start - 5,
        "Water",
        color="w",
        ha="center",
        fontsize=12,
    )
    ax.text(
        config.grid.Nx // 2,
        z_start + skin_points // 2,
        "Skin",
        color="w",
        ha="center",
        fontsize=12,
    )
    ax.text(
        config.grid.Nx // 2,
        z_start + skin_points + skull_points // 2,
        "Skull",
        color="w",
        ha="center",
        fontsize=12,
    )
    ax.text(
        config.grid.Nx // 2,
        z_start + skin_points + skull_points + 10,
        "Brain",
        color="w",
        ha="center",
        fontsize=12,
    )

    # Add transducer array visualization
    source_x_size = config.acoustic.num_elements_x * (
        config.acoustic.pitch / config.grid.dx
    )
    x_start = round((config.grid.Nx - source_x_size) / 2)
    transducer_height = 3
    rect = Rectangle(
        (x_start, 10),
        source_x_size,
        transducer_height,
        facecolor="red",
        alpha=0.5,
        edgecolor="white",
    )
    ax.add_patch(rect)
    ax.text(
        config.grid.Nx // 2,
        8,
        "Transducer Array",
        color="red",
        ha="center",
        fontsize=12,
    )

    return fig, ax


def plot_pressure_field(
    pressure_data: np.ndarray,
    time_step: int,
    config: SimulationConfig,
    slice_y: int | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the pressure field at a specific time step."""
    if slice_y is None:
        slice_y = config.grid.Ny // 2

    fig, ax = plt.subplots(figsize=(12, 8))
    # Take a slice at the specified Y position
    im = ax.imshow(
        pressure_data[time_step, :, slice_y, :].T,
        cmap="coolwarm",
    )
    plt.colorbar(im, label="Pressure [Pa]")
    ax.set_title(f"Pressure Field in Tissue Layers - Time step {time_step}")
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    return fig, ax


def plot_intensity_field(
    intensity_data: np.ndarray,
    config: SimulationConfig,
    slice_y: int | None = None,
    title: str = "Intensity Field",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the intensity field distribution.

    Args:
        intensity_data: 3D array of intensity values (Nx, Ny, Nz)
        config: Simulation configuration
        slice_y: Y-plane to slice (defaults to middle)
        title: Plot title

    Returns:
        Figure and axes objects
    """
    if slice_y is None:
        slice_y = config.grid.Ny // 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        1e-4 * 1e3 * intensity_data[:, slice_y, :].T,  # to mW/cm^2
        cmap="hot",
        aspect="auto",
        norm="log",  # Use log scale for better visualization
    )
    plt.colorbar(im, label="Intensity [mW/cm²]")
    ax.set_title(title)
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    return fig, ax


def make_pressure_video(
    pressure_data: np.ndarray,
    dt: float,
    downsample: int = 4,
    filename: str = "pressure_wave.mp4",
):
    """Make a video of the pressure field."""
    # %%

    # Get global min/max for consistent colorbar
    vmin = pressure_data.min()
    vmax = pressure_data.max()

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pressure_data[0].T, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Pressure [Pa]")
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    def update(frame):
        im.set_array(pressure_data[downsample * frame].T)
        ax.set_title(
            f"Pressure Field in Tissue Layers - {downsample * frame * dt * 1e6} μs"
        )
        return [im]

    nt = pressure_data.shape[0]
    anim = animation.FuncAnimation(
        fig, update, frames=nt // downsample, interval=20, blit=True
    )
    anim.save(filename, writer="ffmpeg")
    plt.close()

    print(f"Video saved as {filename}")
