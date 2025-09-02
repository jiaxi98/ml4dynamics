#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import os



# Default stencil sets for each r
DEFAULT_STENCILS = {
    2: [3, 5, 7, 9],
    4: [5, 7, 9, 11],
    8: [9, 11, 13, 15, 17, 19],
    16: [25, 27, 29, 31, 33, 35],
}

def run_command(cmd, gpus):
    """Run a shell command with CUDA_VISIBLE_DEVICES set."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)




def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to run KS training across r and stencil sizes"
    )
    parser.add_argument(
        "--gpus", type=str, required=True,
        help="CUDA_VISIBLE_DEVICES string, e.g. '0' or '0,1'"
    )
    parser.add_argument(
        "--n", type=int, default=1024,
        help="Fine grid size N1 (stored as 'n' in ks.yaml)"
    )
    # No need for --yaml, Hydra handles config
    parser.add_argument(
        "--include-r16", action="store_true",
        help="Also run r=16 (only meaningful if n=2048)"
    )
    parser.add_argument(
        "--config-name", type=str, default="ks",
        help="Hydra config name to use (e.g. ks, heat1d)"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Also run dataset generation step before training"
    )
    args = parser.parse_args()

    # Which r values to run
    rs = [2, 4, 8]
    if args.include_r16:
        rs.append(16)

    for r in rs:
        stencils = DEFAULT_STENCILS[r]
        for s in stencils:
            print("="*60)
            print(f"Running sweep: N={args.n}, rx={r}, stencil_size={s}, config={args.config_name}")
            print("="*60)
            # Generate dataset if requested
            if args.generate:
                run_command([
                    "python", f"ml4dynamics/dataset_utils/generate_{args.config_name}.py",
                    "--config-name", args.config_name,
                    f"sim.n={args.n}", f"sim.rx={r}", f"sim.stencil_size={s}"
                ], args.gpus)
            # Train model
            run_command([
                "python", "ml4dynamics/trainers/train_jax.py",
                "--config-name", args.config_name,
                f"sim.n={args.n}", f"sim.rx={r}", f"sim.stencil_size={s}"
            ], args.gpus)

if __name__ == "__main__":
    main()
