import argparse
from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import yaml
from box import Box
from jax.scipy.linalg import svd
from matplotlib import pyplot as plt

from ml4dynamics.dataset_utils.dataset_utils import res_int_fn
from ml4dynamics.utils import utils, viz_utils

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None, config_path="../../config", config_name="ks")
def main(cfg: DictConfig):

  # Get logger
  log = logging.getLogger(__name__)

 # parser = argparse.ArgumentParser()
 # parser.add_argument("-c", "--c", default=None, help="constant velocity.")
 # args = parser.parse_args()
 # with open(f"config/ks.yaml", "r") as file:
 #   config_dict = yaml.safe_load(file)
 # config_dict["sim"]["c"] = config_dict["sim"]["c"] if args.c is None\
 #   else float(args.c)

  config_dict = OmegaConf.to_container(cfg, resolve=True)
  config = Box(config_dict)
  c = config.sim.c
  nu = config.sim.nu
  L = config.sim.L
  T = config.sim.T
  dt = config.sim.dt
  BC = config.sim.BC
  if BC == "periodic":
    N1 = config.sim.n
    bc = "pbc"
  elif BC == "Dirichlet-Neumann":
    N1 = config.sim.n - 1
    bc = "dnbc"
  r = config.sim.rx
  N2 = N1 // r
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num
  model_fine, model_coarse = utils.create_fine_coarse_simulator(config)
  res_fn, _ = res_int_fn(config_dict)

  # if sgs_model == "fine_correction":
  #   N = N1
  # else:
  #   N = N2
  N = N2
  inputs = np.zeros((case_num, model_fine.step_num, N))
  outputs_filter = np.zeros((case_num, model_fine.step_num, N))
  outputs_correction = np.zeros((case_num, model_fine.step_num, N))

  # generate spatial coordinates
  if BC == "periodic":
    dx_coarse = L / N2
    x_coords = jnp.linspace(0, L - dx_coarse, N2)
  elif BC == "Dirichlet-Neumann":
    dx_coarse = L / (N2 + 1)
    x_coords = jnp.linspace(dx_coarse, L - dx_coarse, N2)

  x_coords_full = jnp.tile(
    x_coords[None, None, :], (case_num, model_fine.step_num, 1)
  )

  for i in range(case_num):
    print(i)
    rng, key = random.split(rng)

    # fine and coarse-simulations
    # NOTE: the initialization here is important, DO NOT use the random
    # i.i.d. Gaussian noise as the initial condition
    if BC == "periodic":
      dx = model_fine.L / N1
      u0 = model_fine.attractor + model_fine.init_scale * random.normal(key) *\
        jnp.sin(10 * jnp.pi * jnp.linspace(0, L - L/N1, N1) / L)
      r0 = random.uniform(key) * 20 + 44
      u0 = jnp.exp(-(jnp.linspace(0, L - L / N1, N1) - r0)**2 / r0**2 * 4)
      dx_ = L / (N + 1)
      u0_ = jnp.exp(-(jnp.linspace(dx_, L - dx_, N) - r0)**2 / r0**2 * 4)
    elif BC == "Dirichlet-Neumann":
      dx = L / (N1 + 1)
      x = jnp.linspace(dx, L - dx, N1)
      # different choices of initial conditions
      # u0 = model_fine.attractor + init_scale * random.normal(key) *\
      #   jnp.sin(10 * jnp.pi * x / L)
      # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
      #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
      r0 = random.uniform(key) * 20 + 44
      u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
      dx_ = L / (N + 1)
      u0_ = jnp.exp(-(jnp.linspace(dx_, L - dx_, N) - r0)**2 / r0**2 * 4)
    model_fine.run_simulation(u0, model_fine.CN_FEM)
    model_coarse.run_simulation(u0_, model_coarse.CN_FEM)

    # calculating the filter and correction SGS stress
    input = jax.vmap(res_fn)(model_fine.x_hist)[...,
                                                0]  # shape = [step_num, N2]
    output_correction = np.zeros_like(outputs_correction[0])
    output_filter = -(jax.vmap(res_fn)(model_fine.x_hist**2)[..., 0] - input**2) / 2\
      / model_coarse.dx**2
    for j in range(model_fine.step_num):
      next_step_fine = model_fine.CN_FEM(model_fine.x_hist[j])
      next_step_coarse = model_coarse.CN_FEM(input[j])
      output_correction[
        j] = (res_fn(next_step_fine)[:, 0] - next_step_coarse) / dt
      # elif sgs_model == "fine_correction":
      #   output[j] = (next_step_fine - int_fn(next_step_coarse)[:, 0]) / dt

    # if sgs_model == "fine_correction":
    #   inputs[i] = model_fine.x_hist
    inputs[i] = input
    outputs_filter[i] = output_filter
    outputs_correction[i] = output_correction

  # save the data
  inputs = inputs.reshape(-1, N)
  outputs_correction = outputs_correction.reshape(-1, N)
  outputs_filter = outputs_filter.reshape(-1, N)
  x_coords_full = x_coords_full.reshape(-1, N2)
  if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs_filter)) or\
    jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs_filter)) or\
    jnp.any(jnp.isnan(outputs_correction)) or jnp.any(jnp.isinf(outputs_correction)):
    raise Exception("The data contains Inf or NaN")

  data = {
    "metadata": {
      "type": "ks",
      "t0": 0.0,
      "t1": T,
      "dt": dt,
      "n": config.sim.n,
      "BC": BC,
      "description": "Kuramoto–Sivashinsky PDE dataset",
      "author": "Jiaxi Zhao",
      "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "data": {
      "inputs": inputs[..., None],
      "outputs_filter": outputs_filter[..., None],
      "outputs_correction": outputs_correction[..., None],
      "x_coords": x_coords_full[..., None]
    },
    "config":
    config_dict,
    "readme":
    "This dataset contains the results of a Kuramoto–Sivashinsky PDE solver. "
    "The 'input' field represents the velocity, and the 'output' "
    "field represents the correction from the coarse grid simulation."
  }

  r = config.sim.rx
  s = config.sim.stencil_size
  n = config.sim.n
  # Use Hydra's working directory for output
  output_dir = os.path.join(cfg.work_dir, "data/ks")
  os.makedirs(output_dir, exist_ok=True)
  mode_str = "global" if config.train.is_global else "local"
  filename = f"{mode_str}_n{n}_{bc}_nu{nu:.1f}_c{c:.1f}_n{case_num}_r{r}_s{s}.h5"
  filepath = os.path.join(output_dir, filename)
  with h5py.File(filepath, "w") as f:
    metadata_group = f.create_group("metadata")
    for key, value in data["metadata"].items():
      metadata_group.create_dataset(key, data=value)

    data_group = f.create_group("data")
    data_group.create_dataset("inputs", data=data["data"]["inputs"])
    data_group.create_dataset(
      "outputs_filter", data=data["data"]["outputs_filter"]
    )
    data_group.create_dataset(
      "outputs_correction", data=data["data"]["outputs_correction"]
    )
    data_group.create_dataset("x_coords", data=data["data"]["x_coords"])

    config_group = f.create_group("config")
    config_yaml = yaml.dump(data["config"])
    config_group.attrs["config"] = config_yaml

    f.attrs["readme"] = data["readme"]

  plot_ = True
  if plot_ and case_num == 1:
    fig_dir = os.path.join(cfg.work_dir, "results/fig")
    os.makedirs(fig_dir, exist_ok=True)
    t_array = np.linspace(0, T, model_coarse.step_num)
    viz_utils.plot_temporal_corr(
      [inputs, model_coarse.x_hist], [''], t_array, "ks"
    )
    """calculate the commutator of derivative and filter operator"""
    delta1 = jax.vmap(res_fn)(
      jnp.einsum("ij, aj -> ai", model_fine.L1, model_fine.x_hist)
    )[..., 0] - jnp.einsum("ij, aj -> ai", model_coarse.L1, inputs)
    delta2 = jax.vmap(res_fn)(
      jnp.einsum("ij, aj -> ai", model_fine.L2, model_fine.x_hist)
    )[..., 0] - jnp.einsum("ij, aj -> ai", model_coarse.L2, inputs)
    delta4 = jax.vmap(res_fn)(
      jnp.einsum("ij, aj -> ai", model_fine.L4, model_fine.x_hist)
    )[..., 0] - jnp.einsum("ij, aj -> ai", model_coarse.L4, inputs)

    # im_array = np.concatenate(
    #   [inputs.T[None], outputs_filter.T[None], outputs_correction.T[None],
    #   delta1.T[None], delta2.T[None], delta4.T[None]], axis=0
    # )
    outputs_correction = (
      np.roll(outputs_correction, -1, axis=1) -
      np.roll(outputs_correction, 1, axis=1)
    ) / 2 / model_coarse.dx
    im_array = np.concatenate(
      [
        inputs.T[None], outputs_filter.T[None], outputs_correction.T[None],
        outputs_filter.T[None] - outputs_correction.T[None],
        outputs_filter.T[None] + outputs_correction.T[None]
      ],
      axis=0
    )
    utils.plot_with_horizontal_colorbar(
      list(im_array),
      title_array=[
        "u", r"$\\tau^f$", r"$\\tau^c$", r"$[R, D_x]$", r"$[R, D_x^2]$",
        r"$[R, D_x^4]$"
      ],
      shape=(5, 1),
      fig_size=(20, 5),
      file_path=os.path.join(fig_dir, "ks.png"),
      dpi=100
    )

    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    index_array = [500, 1500, 2500, 3500]
    for i in range(len(index_array)):
      axs[i].set_title(f"t = {index_array[i] * config_dict['sim']['dt']:.2f}")
      axs[i].plot(outputs_filter[index_array[i]], label=r"$\\tau^f$")
      axs[i].plot(outputs_correction[index_array[i]], label=r"$\\tau^c$")
      axs[i].plot(delta1[index_array[i]], label=r"$[R, D_x]$")
      axs[i].plot(delta2[index_array[i]], label=r"$[R, D_x^2]$")
      axs[i].plot(delta4[index_array[i]], label=r"$[R, D_x^4]$")
      # axs[i].plot(outputs_filter[index_array[i], :, 0] -\
      #   outputs_correction[index_array[i], :, 0] * ratio, label=r"$\tau^f - \tau^c$")
      axs[i].legend()
    c = config_dict["sim"]["c"]
    plt.savefig(os.path.join(fig_dir, f"cmp_{bc}_nu{nu:.1f}_c{c:.1f}.png"), dpi=300)
    plt.close()
    """visualize the Fourier & POD modes"""
    input_hat = jnp.fft.fft(
      jnp.concatenate([inputs, jnp.zeros((inputs.shape[0], 1))], axis=1),
      axis=1
    )
    input_hat = jnp.fft.fftshift(input_hat, axes=1)
    # i_list = [96, 112, 120, 124, 126, 127, 128, 129, 130, 132, 136, 144, 160]
    i_list = [128, 129, 130, 132, 144]
    _ = plt.figure(figsize=(20, 12))
    for i in i_list:
      plt.plot(input_hat[:, i].real, label=f"Re(k = {i - 128})")
      plt.plot(input_hat[:, i].imag, label=f"Im(k = {i - 128})")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "ks_ode.png"), dpi=300)
    plt.close()

    _, s, vt = svd(
      jnp.array(inputs - jnp.mean(inputs, axis=0)[None]), full_matrices=False
    )
    """NOTE: the singular values decay very slowly for KS equation"""
    s = np.array(s)
    vt = np.array(vt)
    n = 3
    _ = plt.figure(figsize=(12, 12))
    transform_x = inputs @ vt.T
    plt.scatter(transform_x[:, 0], transform_x[:, 1], s=0.1)
    plt.savefig(os.path.join(fig_dir, "pod_ks.png"))
    breakpoint()
    plt.close()


if __name__ == "__main__":
  main()
