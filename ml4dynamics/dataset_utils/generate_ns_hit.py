import argparse
from datetime import datetime
from functools import partial

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax import random
from jax.lax import conv_general_dilated
from matplotlib import pyplot as plt

from ml4dynamics.dataset_utils import dataset_utils
from ml4dynamics.utils import calc_utils, utils, viz_utils


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("-R", "--Re", default=None, help="Reynolds number.")
  args = parser.parse_args()
  with open(f"config/ns_hit.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config_dict["sim"]["Re"] = config_dict["sim"]["Re"] if args.Re is None\
    else int(args.Re)
  config = Box(config_dict)
  T = config.sim.T
  dt = config.sim.dt
  L = config.sim.L
  r = config.sim.rx
  Re = config.sim.Re
  case_num = config.sim.case_num
  rng = random.PRNGKey(config.sim.seed)
  patience = 50
  writeInterval = 1
  model_fine, model_coarse = utils.create_fine_coarse_simulator(config_dict)
  # model_fine.nu = 0
  # model_coarse.nu = 0
  n = model_coarse.N
  inputs = np.zeros((case_num, int(T / dt), n, n, 1))
  outputs_filter = np.zeros((case_num, int(T / dt), n, n, 1))
  outputs_correction = np.zeros((case_num, int(T / dt), n, n, 1))
  print('Generating NS HIT data with n = {}, Re = {} ...'.format(n, Re))

  j = 0
  i = 0
  while j < case_num and i < patience:
    rng, key = random.split(rng)
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))

    # fine and coarse-simulations
    init_cond = "taylor_green" # gaussian_process, real_random, spec_random, taylor_green
    model_fine.w_hat = utils.hit_init_cond(init_cond, model_fine, key)
    model_fine.set_x_hist(model_fine.w_hat, model_fine.CN)
    # model_coarse.w_hat = utils.hit_init_cond(init_cond, model_coarse, key)
    res_fn, _ = dataset_utils.res_int_fn(config_dict)
    w = jnp.fft.irfft2(model_fine.w_hat)
    model_coarse.w_hat = jnp.fft.rfft2(res_fn(w[..., None])[..., 0])
    model_coarse.set_x_hist(model_coarse.w_hat, model_coarse.CN)

    # calculating the filter and correction SGS stress
    # @jax.jit
    def calc_J(what_hist, model):
      psi_hat = -what_hist / model.laplacian_[None]
      dpsidx = np.fft.irfft2(1j * psi_hat * model.kx[None], axes=(1, 2))
      dpsidy = np.fft.irfft2(1j * psi_hat * model.ky[None], axes=(1, 2))
      dwdx = np.fft.irfft2(1j * what_hist * model.kx[None], axes=(1, 2))
      dwdy = np.fft.irfft2(1j * what_hist * model.ky[None], axes=(1, 2))
      return dpsidy * dwdx - dpsidx * dwdy
    kernel_x = kernel_y = r
    kernel = jnp.ones((1, kernel_x, kernel_y, 1)) / kernel_x / kernel_y
    conv = partial(
      conv_general_dilated,
      rhs=kernel,
      window_strides=(r, r),
      padding='VALID',
      dimension_numbers=('NXYC', 'OXYI', 'NXYC'),
    )
    periodic_pad = partial(
      jnp.pad,
      pad_width=(
        (0, 0), (kernel_x // 2, kernel_x // 2 - 1),
        (kernel_y // 2, kernel_y // 2 - 1), (0, 0)
      ),
      mode='wrap'
    )

    if n <= 32:
      """gpu implementation for small size simulation"""
      padded_w = periodic_pad(model_fine.x_hist[..., None])
      w_coarse_ = conv(model_fine.x_hist[..., None])
      w_coarse = jax.vmap(res_fn)(model_fine.x_hist[..., None])
      # assert np.linalg.norm(w_coarse_ - w_coarse) < 1e-14
      J = calc_J(model_fine.xhat_hist, model_fine)
      J_coarse = calc_J(
        jnp.fft.rfft2(w_coarse[..., 0], axes=(1, 2)), model_coarse
      )[..., None]
      padded_J = periodic_pad(J[..., None])
      # J_filter = conv(padded_J)
      J_filter = jax.vmap(res_fn)(J[..., None])
    else:
      """cpu implementation for large size simulation"""

      def res_fn(x):
        result = np.zeros((x.shape[0], n, n, x.shape[-1]))
        for k in range(r):
          for j in range(r):
            result += x[:, k::r, j::r]
        return result / (r**2)

      w_coarse = res_fn(model_fine.x_hist[..., None])
      J = calc_J(model_fine.xhat_hist, model_fine)
      J_coarse = calc_J(
        np.fft.rfft2(w_coarse[..., 0], axes=(1, 2)), model_coarse
      )[..., None]
      J_filter = res_fn(J[..., None])
    # NOTE: be careful with the sign here!!!
    output_filter = -(J_filter - J_coarse) / model_coarse.dx**2

    res_fn, _ = dataset_utils.res_int_fn(config_dict)
    output_correction = np.zeros_like(w_coarse)
    for k in range(model_fine.step_num):
      next_step_fine = model_fine.CN_real(model_fine.x_hist[k][..., None])
      next_step_coarse = model_coarse.CN_real(w_coarse[k])
      output_correction[k] = (res_fn(next_step_fine) - next_step_coarse) / dt

    # output_ = np.zeros_like(w_coarse)
    # for k in range(model_fine.step_num):
    #   next_step_fine = model_fine.CN_real(model_fine.x_hist[k][..., None])
    #   next_step_coarse = model_coarse.CN_real(w_coarse[k])
    #   output_[k] = (res_fn(next_step_fine) - next_step_coarse) / dt

    if not output_filter.shape[1] == output_filter.shape[2] == n:
      breakpoint()
      raise Exception("The shape of output is wrong.")

    if not jnp.isnan(outputs_correction).any() and not jnp.isinf(w_coarse
                                                                 ).any():
      # successful generating traj
      inputs[j] = w_coarse
      outputs_filter[j] = output_filter
      outputs_correction[j] = output_correction
      j = j + 1

  # save the data
  save = True
  if j == case_num and save:
    data = {
      "metadata": {
        "type": "ns",
        "t0": 0.0,
        "t1": T,
        "Re": Re,
        "L": L,
        "dt": dt * writeInterval,
        "nx": n,
        "description": "Navier-Stokes PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "inputs": inputs.reshape(-1, n, n, 1),
        "outputs_filter": outputs_filter.reshape(-1, n, n, 1),
        "outputs_correction": outputs_correction.reshape(-1, n, n, 1),
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Kuramoto–Sivashinsky PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the correction from the coarse grid simulation."
    }

    with h5py.File(f"data/ns_hit/Re{Re}_n{case_num}.h5", "w") as f:
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

      config_group = f.create_group("config")
      config_yaml = yaml.dump(data["config"])
      config_group.attrs["config"] = config_yaml

      f.attrs["readme"] = data["readme"]

  if case_num == 1:

    t_array = np.linspace(0, T, model_coarse.step_num)
    viz_utils.plot_1D_spatial_corr(
      [model_fine.x_hist], [''], "ns_hit"
    )
    viz_utils.plot_temporal_corr(
      [w_coarse[..., 0], model_coarse.x_hist], [''], t_array, "ns_hit"
    )
    breakpoint()
    viz_utils.plot_gif(w_coarse[..., 0], "ns_hit")

    n_plot = 6
    delta = conv(
      jnp.fft.irfft2(
        model_fine.xhat_hist * model_fine.laplacian[None], axes=(1, 2)
      )[..., None]
    ) -\
      jnp.fft.irfft2(
        jnp.fft.rfft2(w_coarse, axes=(1, 2)) * model_coarse.laplacian[None, ..., None],
        axes=(1, 2)
      )
    index_array = np.arange(
      0, n_plot * w_coarse.shape[0] // n_plot - 1, w_coarse.shape[0] // n_plot
    )
    im_array = np.zeros((4, n_plot, *(w_coarse.shape[1:3])))
    for k in range(n_plot):
      im_array[0, k] = w_coarse[index_array[k], ..., 0]
      im_array[1, k] = output_filter[index_array[k], ..., 0]
      im_array[2, k] = output_correction[index_array[k], ..., 0]
      im_array[3, k] = delta[index_array[k], ..., 0]
    im_list = []
    for i in range(4):
      for j in range(n_plot):
        im_list.append(im_array[i, j])
    utils.plot_with_horizontal_colorbar(
      im_list, None, [4, n_plot], (12, 18), "results/fig/dataset.png", 100
    )
    what_coarse = jnp.fft.rfft2(w_coarse, axes=(1, 2))
    psi_hat = -what_coarse[..., 0] / model_coarse.laplacian_[None]
    dpsidx = jnp.fft.irfft2(1j * psi_hat * model_coarse.kx[None], axes=(1, 2))
    dpsidy = jnp.fft.irfft2(1j * psi_hat * model_coarse.ky[None], axes=(1, 2))
    plt.plot(jnp.sum(dpsidx**2 + dpsidy**2, axis=(1, 2)), label='e_kin')
    plt.plot(
      jnp.sum(dpsidx**2 + dpsidy**2, axis=(1, 2))[0] *
      jnp.exp(-model_fine.nu * jnp.linspace(0, model_fine.T,
                                            int(T / dt) + 1)),
      label='decay'
    )
    plt.legend()
    plt.savefig("results/fig/e_kin.png")
    breakpoint()


if __name__ == "__main__":
  main()
