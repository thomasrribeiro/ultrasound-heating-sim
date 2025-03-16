import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import torch
from simulation_config import SimulationConfig


def plot_temperature_evolution(
    times: list, max_temperatures: list, title: str = "Temperature Evolution"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the evolution of maximum temperature over time.

    Args:
        times: List of time points
        max_temperatures: List of maximum temperatures
        title: Plot title

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, max_temperatures)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Maximum Temperature [°C]")
    ax.set_title(title)
    ax.grid(True)

    return fig, ax


def plot_temperature_field_slices(
    temperature_field: torch.Tensor,
    config: SimulationConfig,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Temperature Distribution",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot slices of the 3D temperature field.

    Args:
        temperature_field: 3D temperature field tensor
        config: Simulation configuration
        slice_indices: Tuple of (x, y, z) slice indices (if None, uses middle slices)
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        title: Plot title

    Returns:
        Figure and axes objects
    """
    # Convert to numpy if tensor
    if isinstance(temperature_field, torch.Tensor):
        T_np = temperature_field.cpu().numpy()
    else:
        T_np = temperature_field

    # Get grid parameters from config
    nx, ny, nz = T_np.shape
    dx = config.thermal.dx
    dy = config.thermal.dy
    dz = config.thermal.dz
    Lx = config.thermal.Lx
    Ly = config.thermal.Ly
    Lz = config.thermal.Lz

    # Default to mid-plane indices if not provided
    if slice_indices is None:
        mid_x = nx // 2
        mid_y = ny // 2
        mid_z = nz // 2
    else:
        mid_x, mid_y, mid_z = slice_indices

    # Create figure with 3 subplots for XY, XZ, and YZ planes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Set common colormap limits if not provided
    if vmin is None:
        vmin = T_np.min()
    if vmax is None:
        vmax = T_np.max()

    # 1. XY-plane (Z slice)
    im1 = axes[0].imshow(
        T_np[:, :, mid_z],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"XY Plane (Z={mid_z*dz*1000:.1f}mm)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    plt.colorbar(im1, ax=axes[0], label="Temperature (°C)")

    # 2. XZ-plane (Y slice)
    im2 = axes[1].imshow(
        T_np[:, mid_y, :],
        origin="lower",
        extent=[0, Lx, 0, Lz],
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"XZ Plane (Y={mid_y*dy*1000:.1f}mm)")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Z [m]")
    plt.colorbar(im2, ax=axes[1], label="Temperature (°C)")

    # 3. YZ-plane (X slice)
    im3 = axes[2].imshow(
        T_np[mid_x, :, :],
        origin="lower",
        extent=[0, Ly, 0, Lz],
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title(f"YZ Plane (X={mid_x*dx*1000:.1f}mm)")
    axes[2].set_xlabel("Y [m]")
    axes[2].set_ylabel("Z [m]")
    plt.colorbar(im3, ax=axes[2], label="Temperature (°C)")

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes


def plot_tissue_properties(
    property_field: torch.Tensor,
    config: SimulationConfig,
    property_name: str,
    unit: str,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    colormap: str = "viridis",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot tissue property distribution slices.

    Args:
        property_field: 3D field of tissue property values
        config: Simulation configuration
        property_name: Name of the property being visualized (for labels)
        unit: Unit of the property being visualized
        slice_indices: Tuple of (x, y, z) slice indices (if None, uses middle slices)
        colormap: Matplotlib colormap to use
        title: Plot title (if None, will generate based on property_name)

    Returns:
        Figure and axes objects
    """
    # Convert to numpy if tensor
    if isinstance(property_field, torch.Tensor):
        prop_np = property_field.cpu().numpy()
    else:
        prop_np = property_field

    # Get grid parameters from config
    nx, ny, nz = prop_np.shape
    dx = config.thermal.dx
    dy = config.thermal.dy
    dz = config.thermal.dz
    Lx = config.thermal.Lx
    Ly = config.thermal.Ly
    Lz = config.thermal.Lz

    # Default to mid-plane indices if not provided
    if slice_indices is None:
        mid_x = nx // 2
        mid_y = ny // 2
        mid_z = nz // 2
    else:
        mid_x, mid_y, mid_z = slice_indices

    # Create default title if none provided
    if title is None:
        title = f"{property_name} Distribution"

    # Create figure with 3 subplots for XY, XZ, and YZ planes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. XY-plane (Z slice)
    im1 = axes[0].imshow(
        prop_np[:, :, mid_z],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap=colormap,
    )
    axes[0].set_title(f"XY Plane (Z={mid_z*dz*1000:.1f}mm)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    plt.colorbar(im1, ax=axes[0], label=f"{property_name} [{unit}]")

    # 2. XZ-plane (Y slice)
    im2 = axes[1].imshow(
        prop_np[:, mid_y, :],
        origin="lower",
        extent=[0, Lx, 0, Lz],
        cmap=colormap,
    )
    axes[1].set_title(f"XZ Plane (Y={mid_y*dy*1000:.1f}mm)")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Z [m]")
    plt.colorbar(im2, ax=axes[1], label=f"{property_name} [{unit}]")

    # 3. YZ-plane (X slice)
    im3 = axes[2].imshow(
        prop_np[mid_x, :, :],
        origin="lower",
        extent=[0, Ly, 0, Lz],
        cmap=colormap,
    )
    axes[2].set_title(f"YZ Plane (X={mid_x*dx*1000:.1f}mm)")
    axes[2].set_xlabel("Y [m]")
    axes[2].set_ylabel("Z [m]")
    plt.colorbar(im3, ax=axes[2], label=f"{property_name} [{unit}]")

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes


def plot_temperature_profile(
    temperature_field: torch.Tensor,
    config: SimulationConfig,
    axis: str = "z",
    position: Optional[Tuple[float, float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot temperature profile along a specified axis.

    Args:
        temperature_field: 3D temperature field tensor
        config: Simulation configuration
        axis: Axis along which to plot the profile ('x', 'y', or 'z')
        position: Position coordinates (x, y, z) for the other two axes
                 If None, plots along the center of the domain

    Returns:
        Figure and axes objects
    """
    # Convert to numpy if tensor
    if isinstance(temperature_field, torch.Tensor):
        T_np = temperature_field.cpu().numpy()
    else:
        T_np = temperature_field

    # Get grid parameters from config
    nx, ny, nz = T_np.shape
    dx = config.thermal.dx
    dy = config.thermal.dy
    dz = config.thermal.dz

    # Create coordinate arrays
    x = np.linspace(0, config.thermal.Lx, nx)
    y = np.linspace(0, config.thermal.Ly, ny)
    z = np.linspace(0, config.thermal.Lz, nz)

    # Default to center position if not provided
    if position is None:
        pos_x, pos_y, pos_z = nx // 2, ny // 2, nz // 2
    else:
        # Convert position coordinates to indices
        pos_x = int(min(max(0, position[0] / dx), nx - 1))
        pos_y = int(min(max(0, position[1] / dy), ny - 1))
        pos_z = int(min(max(0, position[2] / dz), nz - 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    if axis.lower() == "x":
        profile = T_np[:, pos_y, pos_z]
        ax.plot(x, profile)
        ax.set_xlabel("X position [m]")
        title = (
            f"Temperature profile along X axis (Y={pos_y*dy:.4f}m, Z={pos_z*dz:.4f}m)"
        )
    elif axis.lower() == "y":
        profile = T_np[pos_x, :, pos_z]
        ax.plot(y, profile)
        ax.set_xlabel("Y position [m]")
        title = (
            f"Temperature profile along Y axis (X={pos_x*dx:.4f}m, Z={pos_z*dz:.4f}m)"
        )
    elif axis.lower() == "z":
        profile = T_np[pos_x, pos_y, :]
        ax.plot(z, profile)
        ax.set_xlabel("Z position [m]")
        title = (
            f"Temperature profile along Z axis (X={pos_x*dx:.4f}m, Y={pos_y*dy:.4f}m)"
        )
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    ax.set_ylabel("Temperature [°C]")
    ax.set_title(title)
    ax.grid(True)

    return fig, ax


def visualize_combined_results(
    acoustic_intensity: torch.Tensor,
    temperature_field: torch.Tensor,
    config: SimulationConfig,
    slice_indices: Optional[Tuple[int, int, int]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create combined visualization of acoustic intensity and temperature.

    Args:
        acoustic_intensity: Acoustic intensity field [W/m²]
        temperature_field: Temperature field [°C]
        config: Simulation configuration
        slice_indices: Tuple of (x, y, z) slice indices (if None, uses middle slices)

    Returns:
        Figure and axes objects
    """
    # Convert to numpy if tensors
    if isinstance(acoustic_intensity, torch.Tensor):
        I_np = acoustic_intensity.cpu().numpy()
    else:
        I_np = acoustic_intensity

    if isinstance(temperature_field, torch.Tensor):
        T_np = temperature_field.cpu().numpy()
    else:
        T_np = temperature_field

    # Get grid parameters
    nx, ny, nz = T_np.shape
    dx = config.thermal.dx
    dy = config.thermal.dy
    dz = config.thermal.dz
    Lx = config.thermal.Lx
    Ly = config.thermal.Ly
    Lz = config.thermal.Lz

    # Default to mid-plane indices if not provided
    if slice_indices is None:
        mid_x = nx // 2
        mid_y = ny // 2
        mid_z = nz // 2
    else:
        mid_x, mid_y, mid_z = slice_indices

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert intensity to mW/cm²
    I_np_mW_cm2 = I_np * 1e-4 * 1e3

    # First row: Acoustic intensity
    # XY plane
    im1 = axes[0, 0].imshow(
        I_np_mW_cm2[:, :, mid_z],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="viridis",
        norm="log",
    )
    axes[0, 0].set_title(f"Intensity: XY Plane (Z={mid_z*dz*1000:.1f}mm)")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    plt.colorbar(im1, ax=axes[0, 0], label="Intensity [mW/cm²]")

    # XZ plane
    im2 = axes[0, 1].imshow(
        I_np_mW_cm2[:, mid_y, :],
        origin="lower",
        extent=[0, Lx, 0, Lz],
        cmap="viridis",
        norm="log",
    )
    axes[0, 1].set_title(f"Intensity: XZ Plane (Y={mid_y*dy*1000:.1f}mm)")
    axes[0, 1].set_xlabel("X [m]")
    axes[0, 1].set_ylabel("Z [m]")
    plt.colorbar(im2, ax=axes[0, 1], label="Intensity [mW/cm²]")

    # YZ plane
    im3 = axes[0, 2].imshow(
        I_np_mW_cm2[mid_x, :, :],
        origin="lower",
        extent=[0, Ly, 0, Lz],
        cmap="viridis",
        norm="log",
    )
    axes[0, 2].set_title(f"Intensity: YZ Plane (X={mid_x*dx*1000:.1f}mm)")
    axes[0, 2].set_xlabel("Y [m]")
    axes[0, 2].set_ylabel("Z [m]")
    plt.colorbar(im3, ax=axes[0, 2], label="Intensity [mW/cm²]")

    # Second row: Temperature
    # Calculate temperature rise from baseline
    T_rise = T_np - config.thermal.arterial_temperature

    # XY plane
    im4 = axes[1, 0].imshow(
        T_rise[:, :, mid_z],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="hot",
    )
    axes[1, 0].set_title(f"Temperature Rise: XY Plane (Z={mid_z*dz*1000:.1f}mm)")
    axes[1, 0].set_xlabel("X [m]")
    axes[1, 0].set_ylabel("Y [m]")
    plt.colorbar(im4, ax=axes[1, 0], label="Temperature Rise [°C]")

    # XZ plane
    im5 = axes[1, 1].imshow(
        T_rise[:, mid_y, :],
        origin="lower",
        extent=[0, Lx, 0, Lz],
        cmap="hot",
    )
    axes[1, 1].set_title(f"Temperature Rise: XZ Plane (Y={mid_y*dy*1000:.1f}mm)")
    axes[1, 1].set_xlabel("X [m]")
    axes[1, 1].set_ylabel("Z [m]")
    plt.colorbar(im5, ax=axes[1, 1], label="Temperature Rise [°C]")

    # YZ plane
    im6 = axes[1, 2].imshow(
        T_rise[mid_x, :, :],
        origin="lower",
        extent=[0, Ly, 0, Lz],
        cmap="hot",
    )
    axes[1, 2].set_title(f"Temperature Rise: YZ Plane (X={mid_x*dx*1000:.1f}mm)")
    axes[1, 2].set_xlabel("Y [m]")
    axes[1, 2].set_ylabel("Z [m]")
    plt.colorbar(im6, ax=axes[1, 2], label="Temperature Rise [°C]")

    # Add overall title
    fig.suptitle("Acoustic Intensity and Temperature Distribution", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig, axes
