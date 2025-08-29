import logging
import os
from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import yaml
from box import Box
from matplotlib import pyplot as plt

from ml4dynamics.dynamics import Heat1D
from ml4dynamics.dataset_utils.dataset_utils import res_int_fn
from ml4dynamics.utils import utils, viz_utils

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../config", config_name="heat1d")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config = Box(config_dict)
    gamma = config.sim.gamma
    L = config.sim.L
    T = config.sim.T
    dt = config.sim.dt
    N1 = config.sim.n
    r = config.sim.rx
    s = config.sim.stencil_size
    bc = "pbc" if config.sim.BC == "periodic" else "dnbc"
    N2 = N1 // r
    seed = config.sim.seed
    case_num = config.sim.case_num

    rng = random.PRNGKey(seed)
    res_fn, _ = res_int_fn(config_dict)

    # Generate spatial coordinates (periodic BC assumed)
    dx_coarse = L / N2
    x_coords = jnp.linspace(0, L - dx_coarse, N2)
    x_coords_full = jnp.tile(x_coords[None, None, :], (case_num, int(T/dt), 1))

    inputs = np.zeros((case_num, int(T/dt), N2))
    outputs_filter = np.zeros((case_num, int(T/dt), N2))
    outputs_correction = np.zeros((case_num, int(T/dt), N2))

    for i in range(case_num):
        print(i)
        rng, key = random.split(rng)
        # Initial condition: Gaussian
        r0 = random.uniform(key) * (L/4) + (L/4)
        u0 = jnp.exp(-(jnp.linspace(0, L - L/N1, N1) - r0)**2 / r0**2 * 4)
        u0_ = jnp.exp(-(jnp.linspace(0, L - L/N2, N2) - r0)**2 / r0**2 * 4)

        # Fine and coarse simulations
        model_fine = Heat1D(L=L, N=N1, T=T, dt=dt, gamma=gamma)
        model_coarse = Heat1D(L=L, N=N2, T=T, dt=dt, gamma=gamma)
        model_fine.run_simulation(u0, model_fine.CN)
        model_coarse.run_simulation(u0_, model_coarse.CN)

        # Filtering and correction
        input = jax.vmap(res_fn)(model_fine.x_hist)[..., 0]
        output_correction = np.zeros_like(outputs_correction[0])
        output_filter = -(jax.vmap(res_fn)(model_fine.x_hist**2)[..., 0] - input**2) / 2 / model_coarse.dx**2
        for j in range(model_fine.step_num):
            next_step_fine = model_fine.CN(model_fine.x_hist[j])
            next_step_coarse = model_coarse.CN(input[j])
            output_correction[j] = (res_fn(next_step_fine)[:, 0] - next_step_coarse) / dt

        inputs[i] = input
        outputs_filter[i] = output_filter
        outputs_correction[i] = output_correction

    # Reshape and check for NaN/Inf
    N = N2
    inputs = inputs.reshape(-1, N)
    outputs_correction = outputs_correction.reshape(-1, N)
    outputs_filter = outputs_filter.reshape(-1, N)
    x_coords_full = x_coords_full.reshape(-1, N)
    if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs_filter)) or\
       jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs_filter)) or\
       jnp.any(jnp.isnan(outputs_correction)) or jnp.any(jnp.isinf(outputs_correction)):
        raise Exception("The data contains Inf or NaN")

    # Save to HDF5
    data = {
        "metadata": {
            "type": "heat1d",
            "t0": 0.0,
            "t1": T,
            "dt": dt,
            "n": N1,
            "description": "1D Heat Equation PDE dataset",
            "author": "Your Name",
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": {
            "inputs": inputs[..., None],
            "outputs_filter": outputs_filter[..., None],
            "outputs_correction": outputs_correction[..., None],
            "x_coords": x_coords_full[..., None]
        },
        "config": config_dict,
        "readme": "This dataset contains the results of a 1D Heat Equation PDE solver."
    }

    output_dir = os.path.join(cfg.work_dir, "data/heat1d")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{bc}_heat1d_gamma{gamma:.2f}_n{case_num}_r{r}_s{s}.h5"
    filepath = os.path.join(output_dir, filename)
    with h5py.File(filepath, "w") as f:
        metadata_group = f.create_group("metadata")
        for key, value in data["metadata"].items():
            metadata_group.create_dataset(key, data=value)

        data_group = f.create_group("data")
        data_group.create_dataset("inputs", data=data["data"]["inputs"])
        data_group.create_dataset("outputs_filter", data=data["data"]["outputs_filter"])
        data_group.create_dataset("outputs_correction", data=data["data"]["outputs_correction"])
        data_group.create_dataset("x_coords", data=data["data"]["x_coords"])

        config_group = f.create_group("config")
        config_yaml = yaml.dump(data["config"])
        config_group.attrs["config"] = config_yaml

        f.attrs["readme"] = data["readme"]

if __name__ == "__main__":
    main()