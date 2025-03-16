import numpy as np
import h5py
from typing import Tuple, Optional
from dataclasses import dataclass

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst

from simulation_config import SimulationConfig


class SkullPressureSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.kgrid: Optional[kWaveGrid] = None
        self.medium: Optional[kWaveMedium] = None
        self.source: Optional[kSource] = None
        self.sensor: Optional[kSensor] = None
        self.sensor_data: Optional[dict] = None

    def setup_grid(self) -> kWaveGrid:
        """Create and configure the k-Wave grid."""
        self.kgrid = kWaveGrid(
            [self.config.Nx, self.config.Ny, self.config.Nz],
            [self.config.dx, self.config.dy, self.config.dz],
        )

        # Calculate time step
        c_max = self.config.skull.sound_speed  # use maximum sound speed for stability
        self.kgrid.makeTime(c_max, t_end=self.config.t_end)

        return self.kgrid

    def setup_medium(self) -> kWaveMedium:
        """Create and configure the layered medium."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before medium")

        # Initialize medium with brain properties
        self.medium = kWaveMedium(
            sound_speed=self.config.brain.sound_speed * np.ones(self.kgrid.k.shape),
            density=self.config.brain.density * np.ones(self.kgrid.k.shape),
            alpha_coeff=self.config.alpha_coeff,
            alpha_power=self.config.alpha_power,
            BonA=self.config.BonA,
        )

        # Convert thicknesses to grid points
        skin_points = round(self.config.skin.thickness / self.config.dx)
        skull_points = round(self.config.skull.thickness / self.config.dx)
        z_start = self.config.initial_tissue_z

        # Add skin layer
        self.medium.sound_speed[:, :, z_start : z_start + skin_points] = (
            self.config.skin.sound_speed
        )
        self.medium.density[:, :, z_start : z_start + skin_points] = (
            self.config.skin.density
        )

        # Add skull layer
        skull_start = z_start + skin_points
        self.medium.sound_speed[:, :, skull_start : skull_start + skull_points] = (
            self.config.skull.sound_speed
        )
        self.medium.density[:, :, skull_start : skull_start + skull_points] = (
            self.config.skull.density
        )

        return self.medium

    def setup_source_sensor(self) -> Tuple[kSource, kSensor]:
        """Create and configure the source and sensor."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before source/sensor")

        # Create time varying source
        source_signal = tone_burst(
            1 / self.kgrid.dt, self.config.freq, self.config.num_cycles
        )
        source_signal = self.config.source_magnitude * source_signal

        # Define source mask for plane wave
        source_x_size = self.config.num_elements_x * (
            self.config.pitch / self.config.dx
        )
        source_y_size = self.config.num_elements_y * (
            self.config.pitch / self.config.dy
        )
        x_start = round((self.config.Nx - source_x_size) / 2)
        y_start = round((self.config.Ny - source_y_size) / 2)

        source_mask = np.zeros(self.kgrid.k.shape)
        source_mask[
            x_start : x_start + int(source_x_size),
            y_start : y_start + int(source_y_size),
            self.config.source_z_pos,
        ] = 1

        # Create source
        self.source = kSource()
        self.source.p_mask = source_mask
        self.source.p = source_signal

        # Create sensor mask and sensor
        sensor_mask = np.zeros(self.kgrid.k.shape)
        sensor_mask[:, :, self.config.initial_tissue_z :] = 1
        self.sensor = kSensor(sensor_mask, record=["p"])

        return self.source, self.sensor

    def run_simulation(self, use_gpu: bool = True) -> dict:
        """Run the k-Wave simulation."""
        if any(x is None for x in [self.kgrid, self.medium, self.source, self.sensor]):
            raise RuntimeError(
                "All simulation components must be set up before running"
            )

        # Set simulation options
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.config.pml_size,
            data_cast="single",
            save_to_disk=True,
            data_recast=True,
            save_to_disk_exit=False,
        )

        # Run simulation
        self.sensor_data = kspaceFirstOrder3D(
            medium=self.medium,
            kgrid=self.kgrid,
            source=self.source,
            sensor=self.sensor,
            simulation_options=simulation_options,
            execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu),
        )

        return self.sensor_data

    def save_results(self, filename: str) -> None:
        """Save simulation results and configuration to HDF5 file."""
        if self.sensor_data is None:
            raise RuntimeError("No simulation data to save")

        with h5py.File(filename, "w") as f:
            # Create groups
            pressure_group = f.create_group("pressure_data")
            grid_group = f.create_group("grid_params")
            time_group = f.create_group("time_params")
            transducer_group = f.create_group("transducer_params")
            layers_group = f.create_group("tissue_layers")

            # Save pressure data
            pressure_group.create_dataset("p", data=self.sensor_data["p"])

            # Save all configuration parameters using the to_dict method
            config_dict = self.config.to_dict()

            # Save grid parameters
            for key, value in config_dict["grid"].items():
                grid_group.attrs[key] = value

            # Save time parameters
            for key, value in config_dict["time"].items():
                time_group.attrs[key] = value

            if hasattr(self.kgrid, "dt"):
                time_group.attrs["dt"] = self.kgrid.dt

            # Save transducer parameters
            for key, value in config_dict["transducer"].items():
                transducer_group.attrs[key] = value

            # Save tissue layer parameters
            for tissue_name, tissue_props in config_dict["tissues"].items():
                tissue_group = layers_group.create_group(tissue_name)
                for key, value in tissue_props.items():
                    tissue_group.attrs[key] = value

            # Save medium properties
            medium_group = f.create_group("medium_properties")
            for key, value in config_dict["medium"].items():
                medium_group.attrs[key] = value

    def compute_intensity(
        self, pressure_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute instantaneous and time-averaged intensity from pressure data.

        Args:
            pressure_data: Pressure data array of shape (time_steps, Nx, Ny, Nz)

        Returns:
            Tuple containing:
            - instantaneous_intensity: Array of shape (time_steps, Nx, Ny, Nz)
            - average_intensity: Array of shape (Nx, Ny, Nz)
        """
        # reset up the medium since it changes for weird reasons
        self.setup_medium()

        avg_pressure_squared = np.mean(pressure_data**2, axis=0)

        density = self.medium.density[
            :, :, self.config.initial_tissue_z :
        ]  # only consider tissue
        sound_speed = self.medium.sound_speed[
            :, :, self.config.initial_tissue_z :
        ]  # only consider tissue

        # Get local acoustic impedance (ρc)
        impedance = density * sound_speed

        # Compute instantaneous intensity I = p²/(2ρc)
        average_intensity_over_simulation = avg_pressure_squared / (2 * impedance)

        # Compute time-averaged intensity
        duty_cycle = self.config.pulse_repetition_freq * self.config.t_end
        average_intensity = average_intensity_over_simulation * duty_cycle

        return average_intensity
