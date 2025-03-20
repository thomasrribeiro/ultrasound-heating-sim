import torch
import numpy as np
import cProfile
import pstats
import time
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.heat.simulator import BioheatSimulator
from src.config import SimulationConfig
import io


def profile_bioheat_simulator():
    """Run profiling on the BioheatSimulator."""
    # Create test configuration
    config = SimulationConfig()
    config.thermal.t_end = 10.0

    # Initialize simulator
    simulator = BioheatSimulator(config)

    # Setup simulator
    print("Setting up mesh...")
    simulator.setup_mesh()

    print("Setting up tissue properties...")
    simulator.setup_tissue_properties()

    # Create a simple intensity field
    print("Creating intensity field...")
    intensity_data = np.load("data/average_intensity.npy")
    intensity = torch.from_numpy(intensity_data).to(simulator.device)

    print("Setting up heat source...")
    simulator.setup_heat_source(intensity_field=intensity)

    # Profiling the simulation
    print("\nProfiling simulation run...")
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    T_history, times, max_temps = simulator.run_simulation()
    elapsed = time.time() - start_time

    pr.disable()

    # Print profiling results
    print(f"\nTotal simulation time: {elapsed:.2f} seconds")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print(s.getvalue())

    # Also profile a single step separately
    print("\nProfiling a single bioheat step...")
    T = (
        torch.ones(
            (config.grid.Nx, config.grid.Ny, config.grid.Nz), device=simulator.device
        )
        * 37.0
    )

    # Warm-up
    simulator.solve_bioheat_step(T, config.thermal.dt)

    # Profile
    pr = cProfile.Profile()
    pr.enable()

    for _ in range(100):  # Repeat for 100 steps to get a better profile
        T = simulator.solve_bioheat_step(T, config.thermal.dt)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    profile_bioheat_simulator()
