import numpy as np
import torch
import torch.jit
import time
from typing import Optional
import h5py

from src.config import SimulationConfig


# Define a standalone JIT-compiled function for bioheat equation solving
@torch.jit.script
def solve_bioheat_step_jit(
    T: torch.Tensor,
    T_padded: torch.Tensor,
    Kt_padded: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    T_a: float,
    inv_dx: float,
    inv_dy: float,
    inv_dz: float,
    dt: float,
) -> torch.Tensor:
    """
    JIT-compiled function to perform one time step of the bioheat equation solver.

    Args:
        T: Current temperature field
        T_padded: Padded temperature field
        Kt_padded: Padded thermal conductivity field
        A: ρ·c term
        B: Blood perfusion term w_b·ρ_b·c_b
        Q: Heat source term
        T_a: Arterial blood temperature
        inv_dx, inv_dy, inv_dz: Inverse of grid spacing
        dt: Time step

    Returns:
        Updated temperature field
    """
    # Compute averaged Kt values at cell faces
    Kt_x_plus = (Kt_padded[2:, 1:-1, 1:-1] + Kt_padded[1:-1, 1:-1, 1:-1]) * 0.5
    Kt_x_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[0:-2, 1:-1, 1:-1]) * 0.5

    Kt_y_plus = (Kt_padded[1:-1, 2:, 1:-1] + Kt_padded[1:-1, 1:-1, 1:-1]) * 0.5
    Kt_y_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[1:-1, 0:-2, 1:-1]) * 0.5

    Kt_z_plus = (Kt_padded[1:-1, 1:-1, 2:] + Kt_padded[1:-1, 1:-1, 1:-1]) * 0.5
    Kt_z_minus = (Kt_padded[1:-1, 1:-1, 1:-1] + Kt_padded[1:-1, 1:-1, 0:-2]) * 0.5

    # Compute temperature gradients
    T_grad_x_plus = (T_padded[2:, 1:-1, 1:-1] - T_padded[1:-1, 1:-1, 1:-1]) * inv_dx
    T_grad_x_minus = (T_padded[1:-1, 1:-1, 1:-1] - T_padded[0:-2, 1:-1, 1:-1]) * inv_dx

    T_grad_y_plus = (T_padded[1:-1, 2:, 1:-1] - T_padded[1:-1, 1:-1, 1:-1]) * inv_dy
    T_grad_y_minus = (T_padded[1:-1, 1:-1, 1:-1] - T_padded[1:-1, 0:-2, 1:-1]) * inv_dy

    T_grad_z_plus = (T_padded[1:-1, 1:-1, 2:] - T_padded[1:-1, 1:-1, 1:-1]) * inv_dz
    T_grad_z_minus = (T_padded[1:-1, 1:-1, 1:-1] - T_padded[1:-1, 1:-1, 0:-2]) * inv_dz

    # Compute fluxes at cell faces
    flux_x_plus = Kt_x_plus * T_grad_x_plus
    flux_x_minus = Kt_x_minus * T_grad_x_minus

    flux_y_plus = Kt_y_plus * T_grad_y_plus
    flux_y_minus = Kt_y_minus * T_grad_y_minus

    flux_z_plus = Kt_z_plus * T_grad_z_plus
    flux_z_minus = Kt_z_minus * T_grad_z_minus

    # Compute divergence of fluxes
    diffusion = (
        (flux_x_plus - flux_x_minus) * inv_dx
        + (flux_y_plus - flux_y_minus) * inv_dy
        + (flux_z_plus - flux_z_minus) * inv_dz
    )

    # Perfusion term: -B * (T - Ta) with spatially-varying B and Ta
    perfusion = -B * (T - T_a)

    # Heat source term
    heat_source = Q

    # Update temperature: T_new = T + dt * (diffusion + perfusion + source) / A with spatially-varying A
    dTdt = (diffusion + perfusion + heat_source) / A
    T_new = T + dt * dTdt

    return T_new


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

        # Pre-allocated tensors for padding
        self._padded_buffer = None
        self._pad_shape = None

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

        # Pre-compute reciprocals for faster computation
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.inv_dz = 1.0 / self.dz

        # Pre-allocate buffer for padded tensor
        self._pad_shape = (self.nx + 2, self.ny + 2, self.nz + 2)
        self._padded_buffer = torch.zeros(self._pad_shape, device=self.device)

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

        # Use the JIT-compiled function for the computation
        T_new = solve_bioheat_step_jit(
            T=T,
            T_padded=T_padded,
            Kt_padded=Kt_padded,
            A=self.A,
            B=self.B,
            Q=self.Q,
            T_a=float(self.T_a),
            inv_dx=float(self.inv_dx),
            inv_dy=float(self.inv_dy),
            inv_dz=float(self.inv_dz),
            dt=float(dt),
        )

        return T_new

    def solve_steady_state_gpu(self, tol=1e-6, max_iter=10000):
        """
        Solve for the steady state temperature field on the GPU via a matrix-free
        conjugate gradient method.

        We solve:

            -div(Kt grad(T)) + B*T = B*T_a + Q

        Args:
            tol: Convergence tolerance for the residual.
            max_iter: Maximum number of CG iterations.

        Returns:
            T: Steady state temperature field as a torch.Tensor of shape (nx, ny, nz).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        device = self.device

        # Precompute factors for finite differences
        idx2 = 1.0 / (self.dx * self.dx)
        idy2 = 1.0 / (self.dy * self.dy)
        idz2 = 1.0 / (self.dz * self.dz)

        # Define the operator function in a matrix-free manner.
        # Given a temperature field T (shape [nx,ny,nz]),
        # it returns L(T) = -div(Kt grad(T)) + B*T.
        def apply_operator(T):
            # Use replicate padding for T and for Kt.
            T_pad = self.pad_3d_replicate(T, pad_width=1)
            K_pad = self.pad_3d_replicate(self.Kt, pad_width=1)

            # x-direction differences
            flux_x_plus = (
                0.5
                * (K_pad[2:, 1:-1, 1:-1] + K_pad[1:-1, 1:-1, 1:-1])
                * (T_pad[2:, 1:-1, 1:-1] - T_pad[1:-1, 1:-1, 1:-1])
                * idx2
            )
            flux_x_minus = (
                0.5
                * (K_pad[1:-1, 1:-1, 1:-1] + K_pad[:-2, 1:-1, 1:-1])
                * (T_pad[1:-1, 1:-1, 1:-1] - T_pad[:-2, 1:-1, 1:-1])
                * idx2
            )

            # y-direction differences
            flux_y_plus = (
                0.5
                * (K_pad[1:-1, 2:, 1:-1] + K_pad[1:-1, 1:-1, 1:-1])
                * (T_pad[1:-1, 2:, 1:-1] - T_pad[1:-1, 1:-1, 1:-1])
                * idy2
            )
            flux_y_minus = (
                0.5
                * (K_pad[1:-1, 1:-1, 1:-1] + K_pad[1:-1, :-2, 1:-1])
                * (T_pad[1:-1, 1:-1, 1:-1] - T_pad[1:-1, :-2, 1:-1])
                * idy2
            )

            # z-direction differences
            flux_z_plus = (
                0.5
                * (K_pad[1:-1, 1:-1, 2:] + K_pad[1:-1, 1:-1, 1:-1])
                * (T_pad[1:-1, 1:-1, 2:] - T_pad[1:-1, 1:-1, 1:-1])
                * idz2
            )
            flux_z_minus = (
                0.5
                * (K_pad[1:-1, 1:-1, 1:-1] + K_pad[1:-1, 1:-1, :-2])
                * (T_pad[1:-1, 1:-1, 1:-1] - T_pad[1:-1, 1:-1, :-2])
                * idz2
            )

            # Divergence: (flux_x_plus - flux_x_minus) + ... etc.
            divergence = (
                flux_x_plus
                - flux_x_minus
                + flux_y_plus
                - flux_y_minus
                + flux_z_plus
                - flux_z_minus
            )

            # The operator L(T) = -divergence + B*T.
            return -divergence + self.B * T

        # Right-hand side of the linear system: b = B*T_a + Q.
        b = self.B * float(self.T_a) + self.Q

        # Initialize the solution with the arterial temperature everywhere.
        T = torch.ones((nx, ny, nz), device=device) * float(self.T_a)

        # Initialize CG variables:
        r = b - apply_operator(T)
        p = r.clone()
        rsold = torch.sum(r * r)

        for it in range(max_iter):
            Ap = apply_operator(p)
            alpha = rsold / torch.sum(p * Ap)
            T = T + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)
            if torch.sqrt(rsnew) < tol:
                print(
                    f"CG converged in {it+1} iterations with residual {torch.sqrt(rsnew):.3e}"
                )
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        else:
            print(
                f"CG did not converge in {max_iter} iterations, residual = {torch.sqrt(rsnew):.3e}"
            )

        return T

    def run_simulation(self, steady_state: bool = False):
        """Run the bioheat simulation.

        Args:
            steady_state: If True, use the steady state solver instead of time stepping.

        Returns:
            Tuple containing:
            - Temperature history tensor (Nt, Nx, Ny, Nz)
            - List of simulation times
            - List of maximum temperatures
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

        # Handle steady state case
        if steady_state:
            print("Using steady state solver...")
            start_time = time.time()

            # Solve for steady state
            T = self.solve_steady_state_gpu()

            # Create minimal history for consistent return format
            times = [0.0]
            max_temps = [float(torch.max(T).cpu().numpy())]
            T_history = np.zeros((1, self.nx, self.ny, self.nz))
            T_history[0] = T.cpu().numpy()

            elapsed = time.time() - start_time
            print(f"Steady state solution completed in {elapsed:.2f} seconds")
            print(f"Max temperature: {max_temps[0]:.6f}°C\n")

            return T_history, times, max_temps

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
