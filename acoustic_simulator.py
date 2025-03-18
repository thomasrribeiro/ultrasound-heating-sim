import numpy as np
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

from config import SimulationConfig


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
            [self.config.grid.Nx, self.config.grid.Ny, self.config.grid.Nz],
            [self.config.grid.dx, self.config.grid.dy, self.config.grid.dz],
        )

        # Calculate time step
        c_max = self.config.skull.sound_speed  # use maximum sound speed for stability
        self.kgrid.makeTime(c_max, t_end=self.config.acoustic.t_end)

        return self.kgrid

    def setup_medium(self) -> kWaveMedium:
        """Create and configure the layered medium."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before medium")

        # Initialize medium with brain properties
        self.medium = kWaveMedium(
            sound_speed=self.config.brain.sound_speed * np.ones(self.kgrid.k.shape),
            density=self.config.brain.density * np.ones(self.kgrid.k.shape),
            alpha_coeff=self.config.acoustic.alpha_coeff,
            alpha_power=self.config.acoustic.alpha_power,
            BonA=self.config.acoustic.BonA,
        )

        # Convert thicknesses to grid points
        skin_points = round(self.config.skin.thickness / self.config.grid.dx)
        skull_points = round(self.config.skull.thickness / self.config.grid.dx)
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
            1 / self.kgrid.dt,
            self.config.acoustic.freq,
            self.config.acoustic.num_cycles,
        )
        source_signal = self.config.acoustic.source_magnitude * source_signal

        # Define source mask for plane wave
        source_x_size = self.config.acoustic.num_elements_x * (
            self.config.acoustic.pitch / self.config.grid.dx
        )
        source_y_size = self.config.acoustic.num_elements_y * (
            self.config.acoustic.pitch / self.config.grid.dy
        )
        x_start = round((self.config.grid.Nx - source_x_size) / 2)
        y_start = round((self.config.grid.Ny - source_y_size) / 2)

        source_mask = np.zeros(self.kgrid.k.shape)
        source_mask[
            x_start : x_start + int(source_x_size),
            y_start : y_start + int(source_y_size),
            self.config.acoustic.source_z_pos,
        ] = 1

        # Create source
        self.source = kSource()
        self.source.p_mask = source_mask
        self.source.p = source_signal

        # Create sensor mask and sensor
        sensor_mask = np.ones(self.kgrid.k.shape)
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
            pml_size=self.config.grid.pml_size,
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

        # Get local acoustic impedance (ρc)
        impedance = self.medium.density * self.medium.sound_speed

        # Compute instantaneous intensity I = p²/(2ρc)
        average_intensity_over_simulation = avg_pressure_squared / (2 * impedance)

        # Compute time-averaged intensity
        duty_cycle = (
            self.config.acoustic.pulse_repetition_freq * self.config.acoustic.t_end
        )
        average_intensity = average_intensity_over_simulation * duty_cycle

        return average_intensity
