"""
Main script for bioheat equation simulation.

This script can either:
1. Run a standalone bioheat simulation with a synthetic Gaussian heat source
2. Use the output from an acoustic simulation to drive the heating
"""

# %% Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

from simulation_config import SimulationConfig
from bioheat_simulator import BioheatSimulator
from bioheat_visualization import (
    plot_temperature_evolution,
    plot_temperature_field_slices,
    plot_tissue_properties,
    plot_temperature_profile,
    visualize_combined_results,
)


def run_standalone_simulation():
    """Run a standalone bioheat simulation with a synthetic heat source."""
    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Create configuration
    config = SimulationConfig()

    # Initialize simulator
    print("Initializing bioheat simulator...")
    simulator = BioheatSimulator(config)

    # Setup mesh
    print("Setting up computational mesh...")
    simulator.setup_mesh()

    # Visualize the layer map
    print("Visualizing tissue layer map...")
    layer_map = simulator.get_layer_map()
    plt.figure(figsize=(10, 5))
    plt.imshow(
        layer_map[:, layer_map.shape[1] // 2, :].cpu().numpy(),
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Tissue Type (0=skin, 1=skull, 2=brain)")
    plt.title("Tissue Layer Map (XZ mid-slice)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.savefig("data/tissue_layer_map.png")
    plt.close()

    # Setup tissue properties
    print("Setting up tissue properties...")
    simulator.setup_tissue_properties()

    # Visualize tissue properties
    print("Visualizing tissue properties...")
    # Thermal conductivity
    fig, _ = plot_tissue_properties(
        simulator.k, config, "Thermal Conductivity", "W/(m·K)", colormap="cool"
    )
    plt.savefig("data/thermal_conductivity_distribution.png")
    plt.close(fig)

    # Absorption coefficient
    fig, ax = plot_tissue_properties(
        simulator.get_absorption_field(),
        config,
        "Acoustic Absorption Coefficient",
        "Np/m",
        colormap="inferno",
    )
    plt.savefig("data/absorption_coefficient_distribution_npm.png")
    plt.close(fig)

    # Absorption coefficient in dB/cm (convert from Np/m)
    absorption_db_cm = simulator.get_absorption_field().cpu().numpy() / 0.1151
    fig, ax = plot_tissue_properties(
        absorption_db_cm,
        config,
        "Acoustic Absorption Coefficient",
        "dB/cm",
        colormap="inferno",
    )
    # Add text annotations for each tissue type
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.1,
        f"Skin: {config.skin.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.2,
        f"Skull: {config.skull.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.3,
        f"Brain: {config.brain.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    plt.savefig("data/absorption_coefficient_distribution_dbcm.png")
    plt.close(fig)

    # Setup heat source
    print("Setting up heat source...")
    simulator.setup_heat_source()

    # Run simulation
    print("Running bioheat simulation...")
    T, times, max_temps = simulator.run_simulation()

    # Plot results
    print("Plotting results...")
    # Temperature evolution
    fig, _ = plot_temperature_evolution(times, max_temps)
    plt.savefig("data/temperature_evolution.png")
    plt.close(fig)

    # Temperature distribution
    fig, _ = plot_temperature_field_slices(T, config)
    plt.savefig("data/temperature_distribution.png")
    plt.close(fig)

    # Temperature profile along z-axis
    fig, _ = plot_temperature_profile(T, config, axis="z")
    plt.savefig("data/temperature_profile_z.png")
    plt.close(fig)

    # Save results
    print("Saving results to HDF5...")
    simulator.save_results("data/bioheat_results.h5")

    print("Simulation complete!")


def run_coupled_simulation(intensity_file):
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

    # Load the acoustic intensity data
    print(f"Loading acoustic intensity data from {intensity_file}...")
    intensity_data = np.load(intensity_file)

    # Create configuration with matching grid size
    config = SimulationConfig()
    # Set thermal grid to match acoustic intensity dimensions
    config.thermal_nx, config.thermal_ny, config.thermal_nz = intensity_data.shape
    # Use the same dx, dy, dz as the acoustic simulation
    config.thermal_dx = config.dx
    config.thermal_dy = config.dy
    config.thermal_dz = config.dz

    # Initialize simulator
    print("Initializing bioheat simulator...")
    simulator = BioheatSimulator(config)

    # Setup mesh
    print("Setting up computational mesh...")
    simulator.setup_mesh()

    # Visualize the layer map
    print("Visualizing tissue layer map...")
    layer_map = simulator.get_layer_map()
    plt.figure(figsize=(10, 5))
    plt.imshow(
        layer_map[:, layer_map.shape[1] // 2, :].cpu().numpy(),
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Tissue Type (0=skin, 1=skull, 2=brain)")
    plt.title("Tissue Layer Map (XZ mid-slice)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.savefig("data/tissue_layer_map_acoustic.png")
    plt.close()

    # Setup tissue properties
    print("Setting up tissue properties...")
    simulator.setup_tissue_properties()

    # Visualize absorption coefficient
    print("Visualizing tissue absorption coefficient...")
    # In Np/m
    fig, _ = plot_tissue_properties(
        simulator.get_absorption_field(),
        config,
        "Acoustic Absorption Coefficient",
        "Np/m",
        colormap="inferno",
    )
    plt.savefig("data/absorption_coefficient_acoustic_npm.png")
    plt.close(fig)

    # In dB/cm
    absorption_db_cm = simulator.get_absorption_field().cpu().numpy() / 0.1151
    fig, ax = plot_tissue_properties(
        absorption_db_cm,
        config,
        "Acoustic Absorption Coefficient",
        "dB/cm",
        colormap="inferno",
    )
    # Add text annotations for each tissue type
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.1,
        f"Skin: {config.skin.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.2,
        f"Skull: {config.skull.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    ax[0].text(
        config.thermal_nx * 0.05,
        config.thermal_ny * 0.3,
        f"Brain: {config.brain.absorption_coefficient_db_cm:.1f} dB/cm",
        color="white",
        fontsize=10,
        ha="left",
    )
    plt.savefig("data/absorption_coefficient_acoustic_dbcm.png")
    plt.close(fig)

    # Setup heat source using acoustic intensity
    print("Setting up heat source from acoustic intensity...")
    intensity_tensor = torch.tensor(intensity_data, device=simulator.device)
    simulator.setup_heat_source(intensity_field=intensity_tensor)

    # Run simulation
    print("Running bioheat simulation...")
    T, times, max_temps = simulator.run_simulation()

    # Plot results
    print("Plotting results...")
    # Temperature evolution
    fig, _ = plot_temperature_evolution(times, max_temps)
    plt.savefig("data/temperature_evolution_acoustic.png")
    plt.close(fig)

    # Temperature distribution
    fig, _ = plot_temperature_field_slices(T, config)
    plt.savefig("data/temperature_distribution_acoustic.png")
    plt.close(fig)

    # Combined acoustic intensity and temperature visualization
    fig, _ = visualize_combined_results(intensity_tensor, T, config)
    plt.savefig("data/acoustic_thermal_combined.png")
    plt.close(fig)

    # Save results
    print("Saving results to HDF5...")
    simulator.save_results("data/bioheat_acoustic_results.h5")

    print("Simulation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bioheat equation simulation")
    parser.add_argument(
        "--intensity",
        type=str,
        help="Path to acoustic intensity .npy file (if not provided, runs standalone simulation)",
    )

    args = parser.parse_args()

    if args.intensity:
        run_coupled_simulation(args.intensity)
    else:
        run_standalone_simulation()
