from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import numpy as np


@dataclass
class TissueProperties:
    # Identity
    name: str  # Name of the tissue type

    # Acoustic properties
    sound_speed: float  # [m/s]
    density: float  # [kg/m^3]
    absorption_coefficient: float  # acoustic absorption coefficient [Np/m]

    # Thermal properties
    specific_heat: float  # specific heat capacity of tissue [J/(kg·K)]
    thermal_conductivity: float  # thermal conductivity [W/(m·K)]
    heat_transfer_rate: float  # heat transfer rate [ml/min/kg]

    thickness: float | None = None  # [m]

    @property
    def blood_perfusion_rate(self) -> float:
        """Blood perfusion rate [1/s]"""
        MIN_TO_SEC = 60
        ML_TO_M3 = 1e-6
        return self.heat_transfer_rate * self.density * ML_TO_M3 / MIN_TO_SEC

    def __post_init__(self):
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
    dx: float = 208e-6  # [m] 208 µm to match transducer pitch
    dy: float = 208e-6  # [m]
    dz: float = 208e-6  # [m]

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
    source_magnitude: float = 0.6e6  # [Pa]
    pulse_repetition_freq: float = 2.7e3  # [Hz]
    pitch: float = 208e-6  # 208 µm pitch between transducer elements

    # Time parameters
    cfl: float = 0.3
    t_end: float = 25e-6  # [s]

    # Medium properties
    alpha_coeff: float = 0.75
    alpha_power: float = 1.5

    # Source and sensor positions
    source_z_pos: int = 10

    # References to other configurations (will be set in SimulationConfig)
    _grid: Optional["GridConfig"] = None
    _tissue_layers: Optional[list["TissueProperties"]] = None

    @property
    def dt(self) -> float:
        """Time step [s]"""
        if self._grid is None or self._tissue_layers is None:
            raise ValueError("_grid and _tissue_layers must be set before accessing dt")
        # Get maximum sound speed across all tissue layers
        max_sound_speed = max(tissue.sound_speed for tissue in self._tissue_layers)
        return self.cfl * self._grid.dx / max_sound_speed


@dataclass
class ThermalConfig:
    """Configuration parameters for the thermal simulation."""

    # Time stepping parameters
    dt: float = 0.01  # time step [s]
    t_end: float = 10.0  # end time [s]
    save_every: int = 100  # save visualization every N steps

    # Blood properties
    blood_density: float = 1050  # [kg/m^3]
    blood_specific_heat: float = 3617  # [J/(kg·K)]
    arterial_temperature: float = 37.0  # [°C]


@dataclass
class SimulationConfig:
    """Complete simulation configuration containing both acoustic and thermal parts."""

    # Tissue layers - ordered from outermost to innermost
    tissue_layers: list[TissueProperties] = field(
        default_factory=lambda: [
            TissueProperties(
                name="gel",
                sound_speed=1624,  # [m/s]
                density=1000,  # [kg/m^3]
                absorption_coefficient=0,  # [Np/m] at 2 MHz
                specific_heat=3630,  # [J/(kg·K)]
                thermal_conductivity=0.51,  # [W/(m·K)] Insulating boundary condition (worst case)
                thickness=4e-3,  # 4 mm
                heat_transfer_rate=559,  # [ml/min/kg]
            ),
            TissueProperties(
                name="skin",
                sound_speed=1624,  # [m/s]
                density=1109,  # [kg/m^3]
                thickness=2e-3,  # 2 mm
                absorption_coefficient=42.3,  # [Np/m] at 2 MHz
                specific_heat=3391,  # [J/(kg·K)]
                thermal_conductivity=0.37,  # [W/(m·K)]
                heat_transfer_rate=106,  # [ml/min/kg]
            ),
            TissueProperties(
                name="skull",
                sound_speed=2770,  # [m/s]
                density=1908,  # [kg/m^3]
                thickness=7e-3,  # 7 mm
                absorption_coefficient=109.1,  # [Np/m] at 2 MHz
                specific_heat=1313,  # [J/(kg·K)]
                thermal_conductivity=0.32,  # [W/(m·K)]
                heat_transfer_rate=10,  # [ml/min/kg]
            ),
            TissueProperties(
                name="brain",
                sound_speed=1546,  # [m/s]
                density=1046,  # [kg/m^3]
                absorption_coefficient=16.75,  # [Np/m] at 2 MHz
                specific_heat=3630,  # [J/(kg·K)]
                thermal_conductivity=0.51,  # [W/(m·K)]
                heat_transfer_rate=559,  # [ml/min/kg]
            ),
        ]
    )

    # Shared grid configuration
    grid: GridConfig = field(default_factory=GridConfig)

    # Separate configurations
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    def __post_init__(self):
        # Compute blood perfusion coefficient for thermal simulation (B = ρ_b·c_b·w_b)
        for tissue in self.tissue_layers:
            tissue.B = (
                self.thermal.blood_density
                * self.thermal.blood_specific_heat
                * tissue.blood_perfusion_rate
            )

        # Set references in acoustic config
        self.acoustic._grid = self.grid
        self.acoustic._tissue_layers = self.tissue_layers

    @property
    def layer_map(self):
        """
        Generate a 3D tissue layer map tensor based on configured tissue thicknesses.

        Returns:
            A tensor with integer values representing tissue types:
            The value corresponds to the index in tissue_layers (0 = first layer, 1 = second layer, etc.)
        """
        # Initialize the layer map with the last tissue type (innermost layer)
        layer_map = (len(self.tissue_layers) - 1) * np.ones(
            (self.grid.Nx, self.grid.Ny, self.grid.Nz),
            dtype=np.long,
        )

        # Start position for first layer
        current_z = 0

        # Assign regions for all layers except the last one (which fills the remaining space)
        for i, tissue in enumerate(self.tissue_layers[:-1]):
            if tissue.thickness is None:
                raise ValueError(f"Tissue layer {i} must have a thickness specified")

            thickness_points = int(tissue.thickness / self.grid.dz)
            if thickness_points > 0:
                next_z = current_z + thickness_points
                layer_map[:, :, current_z:next_z] = i
                current_z = next_z

        return layer_map

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary for saving."""
        return {
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
                },
                "positions": {
                    "source_z": self.acoustic.source_z_pos,
                },
            },
            "thermal": {
                "time": {
                    "dt": self.thermal.dt,
                    "t_end": self.thermal.t_end,
                    "save_every": self.thermal.save_every,
                },
                "blood": {
                    "density": self.thermal.blood_density,
                    "specific_heat": self.thermal.blood_specific_heat,
                    "arterial_temperature": self.thermal.arterial_temperature,
                },
            },
            "tissues": {
                tissue.name: {
                    "acoustic": {
                        "sound_speed": tissue.sound_speed,
                        "density": tissue.density,
                        "thickness": tissue.thickness,
                        "absorption_coefficient_np_m": tissue.absorption_coefficient,
                    },
                    "thermal": {
                        "specific_heat": tissue.specific_heat,
                        "thermal_conductivity": tissue.thermal_conductivity,
                        "blood_perfusion_rate": tissue.blood_perfusion_rate,
                    },
                }
                for tissue in self.tissue_layers
            },
        }
