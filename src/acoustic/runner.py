import os
import numpy as np
import matplotlib.pyplot as plt
from src.acoustic.simulator import PressureSimulator
from src.acoustic.visualization import (
    plot_medium_properties,
    plot_intensity_field,
    make_pressure_video,
)
from src.config import SimulationConfig


def run_acoustic_simulation(
    config: SimulationConfig, output_dir: str, use_gpu: bool = True
) -> np.ndarray:
    """Run the acoustic simulation to generate intensity data."""
    print("\n=== Starting Acoustic Simulation ===")

    # Initialize simulator
    simulator = PressureSimulator(config)

    # Set up the grid
    print("Setting up grid...")
    simulator.setup_grid()

    # Set up the medium
    print("Setting up medium...")
    simulator.setup_medium()

    # Plot the medium properties
    fig, ax = plot_medium_properties(simulator.medium.sound_speed, config)
    plt.savefig(os.path.join(output_dir, "A0_medium_properties.png"))
    plt.close()

    # Set up source and sensor
    print("Setting up source and sensor...")
    simulator.setup_source_sensor()

    # Run the simulation
    print("Running acoustic simulation...")
    sensor_data = simulator.run_simulation(use_gpu=use_gpu)

    # Process and reshape the pressure data
    pressure_data = sensor_data["p"].reshape(
        -1,  # time steps
        config.grid.Nx,
        config.grid.Ny,
        config.grid.Nz,
        order="F",
    )

    # Plot max pressure
    max_pressure = np.max(pressure_data, axis=0)
    plt.figure()
    plt.imshow(1e-6 * max_pressure[:, config.grid.Ny // 2, :].T, cmap="coolwarm")
    plt.colorbar(label="Max Pressure [MPa]")
    plt.title("Max Pressure Field")
    plt.xlabel("X position [grid points]")
    plt.ylabel("Z position [grid points]")
    plt.savefig(os.path.join(output_dir, "A1_max_pressure.png"))
    plt.close()

    # Compute intensity fields
    print("Computing intensity fields...")
    average_intensity = simulator.compute_intensity(pressure_data)

    # Plot time-averaged intensity field
    fig, ax = plot_intensity_field(
        average_intensity,
        config,
        title="Time-Averaged Intensity Field",
    )
    plt.savefig(os.path.join(output_dir, "A2_intensity_field.png"))
    plt.close()

    # Save intensity data
    intensity_path = os.path.join(output_dir, "average_intensity.npy")
    np.save(intensity_path, average_intensity)
    print(f"Saved intensity data to {intensity_path}")

    # make pressure video
    make_pressure_video(
        pressure_data[:, config.grid.Ny // 2, :],
        config.acoustic.dt,
        downsample=10,
        filename=os.path.join(output_dir, "A3_pressure_video.mp4"),
    )

    return average_intensity
