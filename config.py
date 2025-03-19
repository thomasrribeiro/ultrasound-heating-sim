from dataclasses import dataclass, field
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
class GridConfig:
    """Configuration parameters for the simulation grid, shared between acoustic and thermal."""

    # Grid parameters
    domain_size_x: int = 256  # Total domain size before PML subtraction
    domain_size_y: int = 128
    domain_size_z: int = 128

    # PML parameters
    pml_size: int = 10  # Size of the PML layer

    # Derived parameters
    dx: float = 208e-6  # 208 µm to match transducer pitch
    dy: float = 208e-6
    dz: float = 208e-6

    @property
    def Nx(self) -> int:
        """Computational domain size in x direction (excluding PML)"""
        return self.domain_size_x - 2 * self.pml_size

    @property
    def Ny(self) -> int:
        """Computational domain size in y direction (excluding PML)"""
        return self.domain_size_y - 2 * self.pml_size

    @property
    def Nz(self) -> int:
        """Computational domain size in z direction (excluding PML)"""
        return self.domain_size_z - 2 * self.pml_size

    @property
    def Lx(self) -> float:
        """Domain size in x direction [m]"""
        return self.Nx * self.dx

    @property
    def Ly(self) -> float:
        """Domain size in y direction [m]"""
        return self.Ny * self.dy

    @property
    def Lz(self) -> float:
        """Domain size in z direction [m]"""
        return self.Nz * self.dz


@dataclass
class AcousticConfig:
    """Configuration parameters for the acoustic simulation."""

    # Transducer parameters
    freq: float = 2e6  # 2 MHz frequency
    num_cycles: int = 3  # 3 cycle pulse
    num_elements_x: int = 140  # columns
    num_elements_y: int = 64  # rows
    source_magnitude: float = 1e6  # [Pa]
    pulse_repetition_freq: float = 2.7e3  # [Hz]
    pitch: float = 208e-6  # 208 µm pitch between transducer elements

    # Time parameters
    cfl: float = 0.3
    t_end: float = 25e-6  # [s]

    # Medium properties
    alpha_coeff: float = 0.75
    alpha_power: float = 1.5
    BonA: float = 6

    # Source and sensor positions
    source_z_pos: int = 10


@dataclass
class ThermalConfig:
    """Configuration parameters for the thermal simulation."""

    # Time stepping parameters
    dt: float = 0.01  # time step [s]
    steps: int = 500  # number of time steps
    save_every: int = 50  # save visualization every N steps

    # Heat source parameters (will be overridden by acoustic intensity)
    source_magnitude: float = 1e4  # heat source magnitude [W/m^3]
    source_sigma: float = 0.001  # width of Gaussian heat source [m]

    # Blood properties
    blood_density: float = 1000  # [kg/m^3]
    blood_specific_heat: float = 3600  # [J/(kg·K)]
    arterial_temperature: float = 37.0  # [°C]


@dataclass
class SimulationConfig:
    """Complete simulation configuration containing both acoustic and thermal parts."""

    # Tissue layers
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

    # Shared grid configuration
    grid: GridConfig = field(default_factory=GridConfig)

    # Separate configurations
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    # Position where tissue starts in the z-dimension (for acoustic sim)
    initial_tissue_z: int = 20

    def __post_init__(self):
        # Compute blood perfusion coefficient for thermal simulation (B = ρ_b·c_b·w_b)
        self.skin.B = (
            self.thermal.blood_density
            * self.thermal.blood_specific_heat
            * self.skin.blood_perfusion_rate
        )
        self.skull.B = (
            self.thermal.blood_density
            * self.thermal.blood_specific_heat
            * self.skull.blood_perfusion_rate
        )
        self.brain.B = (
            self.thermal.blood_density
            * self.thermal.blood_specific_heat
            * self.brain.blood_perfusion_rate
        )

    @property
    def layer_map(self):
        """
        Generate a 3D tissue layer map tensor based on configured tissue thicknesses.

        Returns:
            A tensor with integer values representing tissue types:
            0 = skin, 1 = skull, 2 = brain
        """
        # Initialize the layer map with all brain tissue (value 2)
        layer_map = 2 * np.ones(
            (self.grid.Nx, self.grid.Ny, self.grid.Nz),
            dtype=np.long,
        )

        # Convert tissue thicknesses to grid points
        skin_thickness_points = int(self.skin.thickness / self.grid.dz)
        skull_thickness_points = int(self.skull.thickness / self.grid.dz)

        # Start positions for layers (assuming layers are along z direction)
        skin_start = self.initial_tissue_z
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
            "initial_tissue_z": self.initial_tissue_z,
            "grid": {
                "domain_size_x": self.grid.domain_size_x,
                "domain_size_y": self.grid.domain_size_y,
                "domain_size_z": self.grid.domain_size_z,
                "pml_size": self.grid.pml_size,
                "Nx": self.grid.Nx,
                "Ny": self.grid.Ny,
                "Nz": self.grid.Nz,
                "dx": self.grid.dx,
                "dy": self.grid.dy,
                "dz": self.grid.dz,
            },
            "acoustic": {
                "transducer": {
                    "frequency": self.acoustic.freq,
                    "num_cycles": self.acoustic.num_cycles,
                    "pitch": self.acoustic.pitch,
                    "num_elements_x": self.acoustic.num_elements_x,
                    "num_elements_y": self.acoustic.num_elements_y,
                    "source_magnitude": self.acoustic.source_magnitude,
                    "pulse_repetition_freq": self.acoustic.pulse_repetition_freq,
                },
                "time": {
                    "cfl": self.acoustic.cfl,
                    "t_end": self.acoustic.t_end,
                },
                "medium": {
                    "alpha_coeff": self.acoustic.alpha_coeff,
                    "alpha_power": self.acoustic.alpha_power,
                    "BonA": self.acoustic.BonA,
                },
                "positions": {
                    "source_z": self.acoustic.source_z_pos,
                },
            },
            "thermal": {
                "time": {
                    "dt": self.thermal.dt,
                    "steps": self.thermal.steps,
                    "save_every": self.thermal.save_every,
                },
                "source": {
                    "magnitude": self.thermal.source_magnitude,
                    "sigma": self.thermal.source_sigma,
                },
                "blood": {
                    "density": self.thermal.blood_density,
                    "specific_heat": self.thermal.blood_specific_heat,
                    "arterial_temperature": self.thermal.arterial_temperature,
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
