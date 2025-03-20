import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Tuple, Optional
import h5py

from src.config import SimulationConfig


class BioheatSimulator:
    """
    Simulator for the bioheat equation with spatially-varying parameters.

    The bioheat equation is:
    ρ·c·∂T/∂t = ∇·(k·∇T) - w_b·ρ_b·c_b·(T - T_a) + Q

    where:
    - ρ: tissue density [kg/m^3]
    - c: specific heat capacity of tissue [J/(kg·K)]
    - T: temperature [°C]
    - t: time [s]
    - k: thermal conductivity [W/(m·K)]
    - w_b: blood perfusion rate [1/s]
    - ρ_b: density of blood [kg/m^3]
    - c_b: specific heat capacity of blood [J/(kg·K)]
    - T_a: arterial blood temperature [°C]
    - Q: heat source [W/m^3]
    """

    def __init__(self, config: SimulationConfig):
        """Initialize the bioheat simulator with configuration parameters."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BioheatSimulator using device: {self.device}")

        # Initialize simulation data
        self.rho = None  # Tissue density field
        self.c = None  # Specific heat capacity field
        self.k = None  # Thermal conductivity field
        self.w_b = None  # Blood perfusion rate field
        self.T_a = None  # Arterial blood temperature field
        self.Q = None  # Heat source field
        self.absorption = None  # Acoustic absorption coefficient field

        # Derived parameters
        self.A = None  # ρ·c
        self.Kt = None  # k
        self.B = None  # w_b·ρ_b·c_b

        # Layer map
        self.layer_map = None

    def setup_mesh(self):
        """Set up the computational grid.

        Returns:
            Initial temperature field tensor
        """
        # Create grid dimensions from config
        self.nx = self.config.grid.Nx
        self.ny = self.config.grid.Ny
        self.nz = self.config.grid.Nz
        self.dx = self.config.grid.dx
        self.dy = self.config.grid.dy
        self.dz = self.config.grid.dz

        # Calculate domain size
        self.Lx = self.config.grid.Lx
        self.Ly = self.config.grid.Ly
        self.Lz = self.config.grid.Lz

        # Create coordinate meshgrid for visualization purposes
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.z = np.linspace(0, self.Lz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # Initialize temperature field with arterial temperature

        # Create coordinate meshgrid on GPU for heat source calculation
        self.X_torch = torch.linspace(0, self.Lx, self.nx, device=self.device)
        self.Y_torch = torch.linspace(0, self.Ly, self.ny, device=self.device)
        self.Z_torch = torch.linspace(0, self.Lz, self.nz, device=self.device)
        self.X_t, self.Y_t, self.Z_t = torch.meshgrid(
            self.X_torch, self.Y_torch, self.Z_torch, indexing="ij"
        )

        self.layer_map = torch.from_numpy(self.config.layer_map).to(self.device)

    def setup_tissue_properties(self):
        """
        Set up spatially-varying tissue properties using the layer map from config.
        """
        # Make sure we have a layer map
        if self.layer_map is None:
            raise RuntimeError("Layer map not initialized. Call setup_mesh() first.")

        # Initialize uniform property fields with default (innermost tissue) values
        self.rho = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.c = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.k = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.w_b = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.absorption = torch.zeros((self.nx, self.ny, self.nz), device=self.device)

        # Initialize arterial temperature as a scalar
        self.T_a = self.config.thermal.arterial_temperature

        # Set properties for each tissue layer based on layer map
        for i, tissue in enumerate(self.config.tissue_layers):
            mask = self.layer_map == i
            self.rho[mask] = tissue.density
            self.c[mask] = tissue.specific_heat
            self.k[mask] = tissue.thermal_conductivity
            self.w_b[mask] = tissue.blood_perfusion_rate
            self.absorption[mask] = tissue.absorption_coefficient

        # Compute derived parameters
        self.A = self.rho * self.c  # [J/(m³·K)]
        self.Kt = self.k  # [W/(m·K)]
        self.B = (
            self.config.thermal.blood_density
            * self.config.thermal.blood_specific_heat
            * self.w_b
        )  # [W/(m³·K)]

        return

    def setup_heat_source(self, intensity_field: Optional[torch.Tensor] = None):
        """
        Set up the heat source term Q.

        Args:
            intensity_field: Optional acoustic intensity field [W/m²].
                            If provided, will convert to volumetric heat source.
                            If None, will create a simple Gaussian heat source.
        """
        if intensity_field is not None:
            # Convert acoustic intensity to heat source using spatially-varying absorption coefficient
            # Q = 2α·I, where α is the absorption coefficient [Np/m]
            # Make sure tissue properties are initialized
            if self.absorption is None:
                raise RuntimeError(
                    "Tissue properties not initialized. Call setup_tissue_properties() first."
                )

            self.Q = 2 * self.absorption * intensity_field
        else:
            raise RuntimeError(
                "Intensity field not provided. Call setup_heat_source() with intensity_field first."
            )
        return self.Q

    def pad_3d_replicate(self, x, pad_width=1):
        """
        Manual implementation of replicate padding for 3D tensor.

        Args:
            x: Input 3D tensor
            pad_width: Width of padding

        Returns:
            Padded tensor
        """
        nx, ny, nz = x.shape
        # Create output tensor with padding
        padded = torch.zeros(
            (nx + 2 * pad_width, ny + 2 * pad_width, nz + 2 * pad_width),
            dtype=x.dtype,
            device=x.device,
        )

        # Copy original data to center
        padded[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width] = x

        # Pad x-direction (left and right faces)
        padded[:pad_width, pad_width:-pad_width, pad_width:-pad_width] = x[0:1].repeat(
            pad_width, 1, 1
        )
        padded[-pad_width:, pad_width:-pad_width, pad_width:-pad_width] = x[-1:].repeat(
            pad_width, 1, 1
        )

        # Pad y-direction (top and bottom faces)
        padded[:, :pad_width, pad_width:-pad_width] = padded[
            :, pad_width : pad_width + 1, pad_width:-pad_width
        ].repeat(1, pad_width, 1)
        padded[:, -pad_width:, pad_width:-pad_width] = padded[
            :, -pad_width - 1 : -pad_width, pad_width:-pad_width
        ].repeat(1, pad_width, 1)

        # Pad z-direction (front and back faces)
        padded[:, :, :pad_width] = padded[:, :, pad_width : pad_width + 1].repeat(
            1, 1, pad_width
        )
        padded[:, :, -pad_width:] = padded[:, :, -pad_width - 1 : -pad_width].repeat(
            1, 1, pad_width
        )

        return padded

    def solve_bioheat_step(self, T, dt):
        """
        Perform one time step of the bioheat equation solver with spatially-varying parameters.

        A * dT/dt = div(Kt * grad(T)) - B * (T - Ta) + Q

        Args:
            T: Current temperature field
            dt: Time step

        Returns:
            Updated temperature field
        """
        # Make padded copy of T for stencil operations
        T_padded = self.pad_3d_replicate(T, pad_width=1)

        # Pad parameter fields for stencil operations
        Kt_padded = self.pad_3d_replicate(self.Kt, pad_width=1)

        # Compute diffusion term div(Kt * grad(T)) with spatially-varying Kt
        # For x-direction: (Kt_{i+1/2,j,k} * (T_{i+1,j,k} - T_{i,j,k}) - Kt_{i-1/2,j,k} * (T_{i,j,k} - T_{i-1,j,k})) / dx²
        # where Kt_{i+1/2,j,k} is approximated as (Kt_{i+1,j,k} + Kt_{i,j,k})/2

        # Compute averaged Kt values at cell faces
        Kt_x_plus = (Kt_padded[2:, 1:-1, 1:-1] + Kt_padded[1:-1, 1:-1, 1:-1]) / 2
        Kt_x_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[0:-2, 1:-1, 1:-1]) / 2

        Kt_y_plus = (Kt_padded[1:-1, 2:, 1:-1] + Kt_padded[1:-1, 1:-1, 1:-1]) / 2
        Kt_y_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[1:-1, 0:-2, 1:-1]) / 2

        Kt_z_plus = (Kt_padded[1:-1, 1:-1, 2:] + Kt_padded[1:-1, 1:-1, 1:-1]) / 2
        Kt_z_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[1:-1, 1:-1, 0:-2]) / 2

        # Compute fluxes at cell faces
        flux_x_plus = (
            Kt_x_plus
            * (T_padded[2:, 1:-1, 1:-1] - T_padded[1:-1, 1:-1, 1:-1])
            / self.dx
        )
        flux_x_minus = (
            Kt_x_minus
            * (T_padded[1:-1, 1:-1, 1:-1] - T_padded[0:-2, 1:-1, 1:-1])
            / self.dx
        )

        flux_y_plus = (
            Kt_y_plus
            * (T_padded[1:-1, 2:, 1:-1] - T_padded[1:-1, 1:-1, 1:-1])
            / self.dy
        )
        flux_y_minus = (
            Kt_y_minus
            * (T_padded[1:-1, 1:-1, 1:-1] - T_padded[1:-1, 0:-2, 1:-1])
            / self.dy
        )

        flux_z_plus = (
            Kt_z_plus
            * (T_padded[1:-1, 1:-1, 2:] - T_padded[1:-1, 1:-1, 1:-1])
            / self.dz
        )
        flux_z_minus = (
            Kt_z_minus
            * (T_padded[1:-1, 1:-1, 1:-1] - T_padded[1:-1, 1:-1, 0:-2])
            / self.dz
        )

        # Compute divergence of fluxes
        diffusion = (
            (flux_x_plus - flux_x_minus) / self.dx
            + (flux_y_plus - flux_y_minus) / self.dy
            + (flux_z_plus - flux_z_minus) / self.dz
        )

        # Perfusion term: -B * (T - Ta) with spatially-varying B and Ta
        perfusion = -self.B * (T - self.T_a)

        # Heat source term
        heat_source = self.Q

        # Update temperature: T_new = T + dt * (diffusion + perfusion + source) / A with spatially-varying A
        dTdt = (diffusion + perfusion + heat_source) / self.A
        T_new = T + dt * dTdt

        return T_new

    def run_simulation(self):
        """Run the bioheat simulation.

        Returns:
            Tuple containing:
            - Final temperature field tensor (Nx, Ny, Nz)
            - List of simulation times
            - List of maximum temperatures
            - Temperature history tensor (Nt, Nx, Ny, Nz)
        """
        # Check if setup has been completed
        if self.A is None or self.Kt is None or self.B is None or self.Q is None:
            raise RuntimeError(
                "Simulation setup is incomplete. Call setup_mesh(), setup_tissue_properties(), and setup_heat_source() first."
            )

        # Get time stepping parameters from config
        dt = self.config.thermal.dt
        t_end = self.config.thermal.t_end
        save_every = self.config.thermal.save_every

        # Calculate number of steps
        steps = int(t_end / dt)

        # Initialize time and history arrays
        start_time = time.time()
        t = 0.0
        times = []
        max_temps = []
        # Calculate number of history points to save
        n_history = len(range(0, steps, save_every)) + 1  # +1 for final step
        T_history = np.zeros((n_history, self.nx, self.ny, self.nz))
        history_idx = 0

        # Initialize temperature field
        T = (
            torch.ones((self.nx, self.ny, self.nz), device=self.device)
            * self.config.thermal.arterial_temperature
        )

        # Time stepping loop
        for step in range(steps):
            t += dt

            # Update temperature
            T = self.solve_bioheat_step(T, dt)

            # Save and output statistics
            if step % save_every == 0 or step == steps - 1:
                # Get max temperature
                max_temp = float(torch.max(T).cpu().numpy())
                times.append(t)
                max_temps.append(max_temp)
                T_history[history_idx] = T.cpu().numpy()  # Store temperature field
                history_idx += 1

                print(
                    f"Step {step}/{steps}, Time: {t:.3f}s, Max temp: {max_temp:.6f}°C"
                )

        # Calculate performance
        elapsed = time.time() - start_time
        print(
            f"Simulation completed in {elapsed:.2f} seconds ({steps/elapsed:.1f} steps/second)"
        )

        return T_history, times, max_temps

    def get_layer_map(self):
        """Get the tissue layer map."""
        if self.layer_map is None:
            raise RuntimeError("Layer map not initialized")
        return self.layer_map

    def get_absorption_field(self):
        """Get the acoustic absorption coefficient field."""
        if self.absorption is None:
            raise RuntimeError("Absorption field not initialized")
        return self.absorption

    def save_results(
        self,
        filename: str,
        T: torch.Tensor,
        times: list,
        max_temps: list,
        T_history: torch.Tensor,
    ) -> None:
        """
        Save simulation results to an HDF5 file.

        Args:
            filename: Output file name
            T: Final temperature field
            times: List of simulation times
            max_temps: List of maximum temperatures
            T_history: Temperature history tensor (Nt, Nx, Ny, Nz)
        """
        # Transfer temperature field to CPU
        T_np = T.cpu().numpy()

        with h5py.File(filename, "w") as f:
            # Create groups
            temp_group = f.create_group("temperature_data")
            time_group = f.create_group("time_params")
            source_group = f.create_group("source_params")
            tissues_group = f.create_group("tissue_properties")

            # Save temperature data
            temp_group.create_dataset("T", data=T_np)
            temp_group.create_dataset("history_times", data=np.array(times))
            temp_group.create_dataset("history_max_temp", data=np.array(max_temps))
            temp_group.create_dataset("T_history", data=T_history.cpu().numpy())

            # Save layer map if available
            if self.layer_map is not None:
                temp_group.create_dataset(
                    "layer_map", data=self.layer_map.cpu().numpy()
                )

            # Save absorption field if available
            if self.absorption is not None:
                temp_group.create_dataset(
                    "absorption", data=self.absorption.cpu().numpy()
                )

            # Save thermal config parameters using the to_dict method
            config_dict = self.config.to_dict()

            # Save time parameters
            for key, value in config_dict["thermal"]["time"].items():
                time_group.attrs[key] = value

            # Save source parameters
            for key, value in config_dict["thermal"]["source"].items():
                source_group.attrs[key] = value

            # Save tissue properties for each layer
            for tissue in self.config.tissue_layers:
                tissue_group = tissues_group.create_group(tissue.name)
                # Save thermal properties
                for key, value in config_dict["tissues"][tissue.name][
                    "thermal"
                ].items():
                    tissue_group.attrs[key] = value
                # Save acoustic properties relevant to heating
                tissue_group.attrs["absorption_coefficient"] = (
                    tissue.absorption_coefficient
                )

            # Save blood properties
            blood_group = tissues_group.create_group("blood")
            for key, value in config_dict["thermal"]["blood"].items():
                blood_group.attrs[key] = value
