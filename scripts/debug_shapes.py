#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import yaml

import jax
import jax.numpy as jnp
from box import Box

from ml4dynamics.utils import utils

jax.config.update("jax_enable_x64", True)


def find_ckpt(pde="ks"):
    p = f"ckpts/{pde}"
    if not os.path.isdir(p):
        return None
    files = sorted(glob.glob(os.path.join(p, "*.pkl")), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pkl (optional)")
    parser.add_argument("--config", type=str, default="config/ks.yaml", help="Path to YAML config")
    parser.add_argument("--n", type=int, default=None, help="override sim.n")
    parser.add_argument("--rx", type=int, default=None, help="override sim.rx")
    parser.add_argument("--stencil_size", type=int, default=None, help="override sim.stencil_size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # apply overrides if provided
    if args.n:
        cfg.setdefault("sim", {})["n"] = args.n
    if args.rx:
        cfg.setdefault("sim", {})["rx"] = args.rx
    if args.stencil_size:
        cfg.setdefault("sim", {})["stencil_size"] = args.stencil_size

    print("Using config:")
    print(yaml.safe_dump(cfg["sim"]))

    ckpt = args.ckpt or find_ckpt(cfg.get("case", "ks"))
    if ckpt is None:
        print("No checkpoint found in ckpts/, pass --ckpt to specify one.")
        return
    print("Using checkpoint:", ckpt)

    # create simulators
    sim_fine, sim_coarse = utils.create_fine_coarse_simulator(cfg)
    N_fine = sim_fine.N if hasattr(sim_fine, "N") else getattr(sim_fine, "n", None)
    N_coarse = sim_coarse.N if hasattr(sim_coarse, "N") else getattr(sim_coarse, "n", None)
    print(f"Sim sizes: fine N={N_fine}, coarse N={N_coarse}")

    # prepare train state and load params from checkpoint
    is_global = cfg.get("train", {}).get("is_global", True)
    train_state, _ = utils.prepare_unet_train_state(cfg, load_dict=ckpt, is_global=is_global, is_training=False)

    # Build a sample input (as used in run_simulation_coarse_grid_correction)
    # For KS local model, x has shape (N, 1); we pass batch dim 1 -> (1, N, 1)
    N = N_fine if is_global else (N_fine if cfg["sim"].get("BC", "periodic") == "periodic" else N_fine)
    x_sample = jnp.zeros((1, N, 1), dtype=jnp.float64)
    print("x_sample.shape:", x_sample.shape)

    # For local model, forward_fn in train_jax uses augment_inputs then reshape
    if not is_global:
        stencil_size = cfg["sim"].get("stencil_size", 3)
        input_features = len(cfg["train"].get("input_features", ["u"])) * stencil_size
        print("Expected local input_features (flattened):", input_features)
        # augment_inputs will not use x_coords unless 'x' is in input_features; pass dummy
        x_coords = jnp.zeros((1, N, 1))
        x_aug = utils.augment_inputs(x_sample, x_coords, cfg.get("case", "ks"), cfg["train"].get("input_features", ["u"]), stencil_size, sim_fine)
        print("augmented.shape:", x_aug.shape)
        x_flat = x_aug.reshape(-1, x_aug.shape[-1])
        print("flattened for model input shape:", x_flat.shape)
        out = train_state.apply_fn(train_state.params, x_flat)
        print("raw model output shape (flattened):", out.shape)
        out_reshaped = out.reshape(x_sample.shape)
        print("model output reshaped to state shape:", out_reshaped.shape)
    else:
        # global UNet: expects input shape (1, N, channels)
        # prepare a dummy input consistent with prepare_unet_train_state init
        channels = 1
        x_in = jnp.zeros((1, N, channels), dtype=jnp.float64)
        print("global input shape:", x_in.shape)
        out = train_state.apply_fn_with_bn({"params": train_state.params, "batch_stats": train_state.batch_stats}, x_in, is_training=False)[0]
        print("global model raw output shape:", out.shape)

    print("Done.")


if __name__ == '__main__':
    main()
