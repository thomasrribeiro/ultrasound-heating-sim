from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import numpy as np


@dataclass
class TissueProperties:
    # Acoustic properties
    sound_speed: float  # [m/s]
    density: float  # [kg/m^3]
    thickness: float = 0.0  # [m]
    absorption_coefficient_db_cm: float = 1.0  # acoustic absorption coefficient [dB/cm]

    # Thermal properties
    specific_heat: float = 3700  # specific heat capacity of tissue [J/(kg·K)]
    thermal_conductivity: float = 0.5  # thermal conductivity [W/(m·K)]
    blood_perfusion_rate: float = 0.0005  # blood perfusion rate [1/s]

    def __post_init__(self):
        # Convert absorption coefficient from dB/cm to Np/m
        self.absorption_coefficient = (
            self.absorption_coefficient_db_cm * 0.1151
        )  # [Np/m]

        # Derived thermal parameters
        self.A = self.density * self.specific_heat  # [J/(m³·K)]
        self.Kt = self.thermal_conductivity  # [W/(m·K)]


@dataclass
class SimulationConfig:
    # --------------------------#
    # ACOUSTIC SIMULATION PART #
    # --------------------------#

    # Transducer parameters
    freq: float = 2e6  # 2 MHz frequency
    num_cycles: int = 3  # 3 cycle pulse
    pitch: float = 208e-6  # 208 µm pitch
    num_elements_x: int = 140  # columns
    num_elements_y: int = 64  # rows
    source_magnitude: float = 1e6  # [Pa]
    pulse_repetition_freq: float = 2.7e3  # [Hz]

    # Acoustic grid parameters
    dx: float = pitch  # spatial step [m]
    dy: float = pitch
    dz: float = pitch
    pml_size: int = 10
    Nx: int = 256 - 2 * pml_size
    Ny: int = 128 - 2 * pml_size
    Nz: int = 128 - 2 * pml_size

    # Acoustic time parameters
    cfl: float = 0.3
    t_end: float = 25e-6  # [s]

    # Acoustic medium properties
    alpha_coeff: float = 0.75
    alpha_power: float = 1.5
    BonA: float = 6

    # Source and sensor positions
    source_z_pos: int = 10
    initial_tissue_z: int = 20

    # Blood properties (shared between acoustic and thermal parts)
    blood_density: float = 1000  # [kg/m^3]
    blood_specific_heat: float = 3600  # [J/(kg·K)]
    arterial_temperature: float = 37.0  # [°C]

    # Tissue layers (acoustic properties)
    skin: TissueProperties = TissueProperties(
        sound_speed=1610,  # [m/s]
        density=1090,  # [kg/m^3]
        thickness=2e-3,  # 2 mm
        absorption_coefficient_db_cm=1.5,  # [dB/cm] at 2 MHz
        specific_heat=3500,  # [J/(kg·K)]
        thermal_conductivity=0.42,  # [W/(m·K)]
        blood_perfusion_rate=0.002,  # [1/s]
    )

    skull: TissueProperties = TissueProperties(
        sound_speed=3200,  # [m/s]
        density=1900,  # [kg/m^3]
        thickness=7e-3,  # 7 mm
        absorption_coefficient_db_cm=7.0,  # [dB/cm] at 2 MHz
        specific_heat=1300,  # [J/(kg·K)]
        thermal_conductivity=0.32,  # [W/(m·K)]
        blood_perfusion_rate=0.0003,  # [1/s]
    )

    brain: TissueProperties = TissueProperties(
        sound_speed=1560,  # [m/s]
        density=1040,  # [kg/m^3]
        absorption_coefficient_db_cm=1.0,  # [dB/cm] at 2 MHz
        specific_heat=3600,  # [J/(kg·K)]
        thermal_conductivity=0.51,  # [W/(m·K)]
        blood_perfusion_rate=0.008,  # [1/s]
    )

    # -------------------------#
    # THERMAL SIMULATION PART #
    # -------------------------#

    # Thermal grid parameters (can be different from acoustic grid)
    thermal_use_same_grid: bool = False  # If True, use same grid as acoustic sim
    thermal_nx: int = 30  # number of cells in x direction
    thermal_ny: int = 30  # number of cells in y direction
    thermal_nz: int = 30  # number of cells in z direction
    thermal_dx: float = 0.001  # cell size in x direction [m]
    thermal_dy: float = 0.001  # cell size in y direction [m]
    thermal_dz: float = 0.001  # cell size in z direction [m]

    # Thermal time stepping parameters
    thermal_dt: float = 0.01  # time step [s]
    thermal_steps: int = 500  # number of time steps
    thermal_save_every: int = 50  # save visualization every N steps

    # Thermal heat source parameters (will be overridden by acoustic intensity)
    thermal_source_magnitude: float = 1e4  # heat source magnitude [W/m^3]
    thermal_source_sigma: float = 0.001  # width of Gaussian heat source [m]

    def __post_init__(self):
        # Calculate derived parameters
        if self.thermal_use_same_grid:
            # Use the same grid as the acoustic simulation
            self.thermal_nx = self.Nx
            self.thermal_ny = self.Ny
            self.thermal_nz = self.Nz
            self.thermal_dx = self.dx
            self.thermal_dy = self.dy
            self.thermal_dz = self.dz

        # Derived thermal grid dimensions
        self.thermal_Lx = self.thermal_nx * self.thermal_dx
        self.thermal_Ly = self.thermal_ny * self.thermal_dy
        self.thermal_Lz = self.thermal_nz * self.thermal_dz

        # Compute blood perfusion coefficient for thermal simulation (B = ρ_b·c_b·w_b)
        self.skin.B = (
            self.blood_density
            * self.blood_specific_heat
            * self.skin.blood_perfusion_rate
        )
        self.skull.B = (
            self.blood_density
            * self.blood_specific_heat
            * self.skull.blood_perfusion_rate
        )
        self.brain.B = (
            self.blood_density
            * self.blood_specific_heat
            * self.brain.blood_perfusion_rate
        )

    def generate_layer_map(self, device):
        """
        Generate a 3D tissue layer map tensor based on configured tissue thicknesses.

        Args:
            device: The torch device to create the tensor on

        Returns:
            A tensor with integer values representing tissue types:
            0 = skin, 1 = skull, 2 = brain
        """
        # Initialize the layer map with all brain tissue (value 2)
        layer_map = (
            torch.ones(
                (self.thermal_nx, self.thermal_ny, self.thermal_nz),
                dtype=torch.long,
                device=device,
            )
            * 2
        )

        # Convert tissue thicknesses to grid points
        skin_thickness_points = int(self.skin.thickness / self.thermal_dz)
        skull_thickness_points = int(self.skull.thickness / self.thermal_dz)

        # Start positions for layers (assuming layers are along z direction)
        skin_start = 0  # Start at the top of the domain
        skull_start = skin_start + skin_thickness_points
        brain_start = skull_start + skull_thickness_points

        # Assign skin (0) and skull (1) regions
        if skin_thickness_points > 0:
            layer_map[:, :, skin_start:skull_start] = 0  # Skin

        if skull_thickness_points > 0:
            layer_map[:, :, skull_start:brain_start] = 1  # Skull

        return layer_map

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary for saving."""
        return {
            "acoustic": {
                "transducer": {
                    "frequency": self.freq,
                    "num_cycles": self.num_cycles,
                    "pitch": self.pitch,
                    "num_elements_x": self.num_elements_x,
                    "num_elements_y": self.num_elements_y,
                    "source_magnitude": self.source_magnitude,
                    "pulse_repetition_freq": self.pulse_repetition_freq,
                },
                "grid": {
                    "dx": self.dx,
                    "dy": self.dy,
                    "dz": self.dz,
                    "Nx": self.Nx,
                    "Ny": self.Ny,
                    "Nz": self.Nz,
                    "pml_size": self.pml_size,
                },
                "time": {
                    "cfl": self.cfl,
                    "t_end": self.t_end,
                },
                "medium": {
                    "alpha_coeff": self.alpha_coeff,
                    "alpha_power": self.alpha_power,
                    "BonA": self.BonA,
                },
                "positions": {
                    "source_z": self.source_z_pos,
                    "initial_tissue_z": self.initial_tissue_z,
                },
            },
            "thermal": {
                "grid": {
                    "nx": self.thermal_nx,
                    "ny": self.thermal_ny,
                    "nz": self.thermal_nz,
                    "dx": self.thermal_dx,
                    "dy": self.thermal_dy,
                    "dz": self.thermal_dz,
                },
                "time": {
                    "dt": self.thermal_dt,
                    "steps": self.thermal_steps,
                    "save_every": self.thermal_save_every,
                },
                "source": {
                    "magnitude": self.thermal_source_magnitude,
                    "sigma": self.thermal_source_sigma,
                },
                "blood": {
                    "density": self.blood_density,
                    "specific_heat": self.blood_specific_heat,
                    "arterial_temperature": self.arterial_temperature,
                },
            },
            "tissues": {
                "skin": {
                    "acoustic": {
                        "sound_speed": self.skin.sound_speed,
                        "density": self.skin.density,
                        "thickness": self.skin.thickness,
                        "absorption_coefficient_db_cm": self.skin.absorption_coefficient_db_cm,
                        "absorption_coefficient_np_m": self.skin.absorption_coefficient,
                    },
                    "thermal": {
                        "specific_heat": self.skin.specific_heat,
                        "thermal_conductivity": self.skin.thermal_conductivity,
                        "blood_perfusion_rate": self.skin.blood_perfusion_rate,
                    },
                },
                "skull": {
                    "acoustic": {
                        "sound_speed": self.skull.sound_speed,
                        "density": self.skull.density,
                        "thickness": self.skull.thickness,
                        "absorption_coefficient_db_cm": self.skull.absorption_coefficient_db_cm,
                        "absorption_coefficient_np_m": self.skull.absorption_coefficient,
                    },
                    "thermal": {
                        "specific_heat": self.skull.specific_heat,
                        "thermal_conductivity": self.skull.thermal_conductivity,
                        "blood_perfusion_rate": self.skull.blood_perfusion_rate,
                    },
                },
                "brain": {
                    "acoustic": {
                        "sound_speed": self.brain.sound_speed,
                        "density": self.brain.density,
                        "absorption_coefficient_db_cm": self.brain.absorption_coefficient_db_cm,
                        "absorption_coefficient_np_m": self.brain.absorption_coefficient,
                    },
                    "thermal": {
                        "specific_heat": self.brain.specific_heat,
                        "thermal_conductivity": self.brain.thermal_conductivity,
                        "blood_perfusion_rate": self.brain.blood_perfusion_rate,
                    },
                },
            },
        }
