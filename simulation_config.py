from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TissueProperties:
    sound_speed: float  # [m/s]
    density: float  # [kg/m^3]
    thickness: float = 0.0  # [m]


@dataclass
class SimulationConfig:
    # Transducer parameters
    freq: float = 2e6  # 2 MHz frequency
    num_cycles: int = 3  # 3 cycle pulse
    pitch: float = 208e-6  # 208 µm pitch
    num_elements_x: int = 140  # columns
    num_elements_y: int = 64  # rows
    source_magnitude: float = 1e6  # [Pa]
    pulse_repetition_freq: float = 2.7e3  # [Hz]

    # Grid parameters
    dx: float = pitch  # spatial step [m]
    dy: float = pitch
    dz: float = pitch
    pml_size: int = 10
    Nx: int = 256 - 2 * pml_size
    Ny: int = 128 - 2 * pml_size
    Nz: int = 128 - 2 * pml_size

    # Time parameters
    cfl: float = 0.3
    t_end: float = 25e-6  # [s]

    # Medium properties
    alpha_coeff: float = 0.75
    alpha_power: float = 1.5
    BonA: float = 6

    # Source and sensor positions
    source_z_pos: int = 10
    initial_tissue_z: int = 20

    # Tissue layers
    skin: TissueProperties = TissueProperties(
        sound_speed=1610,  # [m/s]
        density=1090,  # [kg/m^3]
        thickness=2e-3,  # 2 mm
    )

    skull: TissueProperties = TissueProperties(
        sound_speed=3200,  # [m/s]
        density=1900,  # [kg/m^3]
        thickness=7e-3,  # 7 mm
    )

    brain: TissueProperties = TissueProperties(
        sound_speed=1560,  # [m/s]
        density=1040,  # [kg/m^3]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary for saving."""
        return {
            "transducer": {
                "frequency": self.freq,
                "num_cycles": self.num_cycles,
                "pitch": self.pitch,
                "num_elements_x": self.num_elements_x,
                "num_elements_y": self.num_elements_y,
                "source_magnitude": self.source_magnitude,
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
            "tissues": {
                "skin": vars(self.skin),
                "skull": vars(self.skull),
                "brain": vars(self.brain),
            },
        }
