#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path

from src.heat.runner import run_heat_simulation
from src.acoustic.runner import run_acoustic_simulation
from src.config import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="Run acoustic and/or heat simulations")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use CPU for computations (default: False). Use --use_cpu to enable.",
    )
    parser.add_argument(
        "--intensity-file",
        type=str,
        help="Path to pre-computed intensity data (.npy file). If not provided, acoustic simulation will be run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save output files (default: data)",
    )
    parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Use steady state solver for heat simulation instead of time stepping",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize configuration
    config = SimulationConfig()

    # Get intensity data
    if args.intensity_file:
        # Use pre-computed intensity
        intensity_path = Path(args.intensity_file)
        if not intensity_path.exists():
            raise FileNotFoundError(f"Intensity file not found: {args.intensity_file}")
        print(f"Loading pre-computed intensity data from {args.intensity_file}")
        intensity_data = np.load(args.intensity_file)
    else:
        # Run acoustic simulation
        intensity_data = run_acoustic_simulation(
            config, args.output_dir, use_gpu=not args.use_cpu
        )

    # Run heat simulation
    run_heat_simulation(
        config, intensity_data, args.output_dir, steady_state=args.steady_state
    )


if __name__ == "__main__":
    main()
