import gc
import math
import pickle
from functools import partial
from time import time

import h5py
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import ml_collections
import numpy as np
import optax
import torch
from box import Box
from flax import traverse_util
from flax.training.train_state import TrainState
from jax import random as random
from jax.numpy.linalg import solve
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ml4dynamics import dynamics
from ml4dynamics.dataset_utils.dataset_utils import res_int_fn
from ml4dynamics.models.models_jax import MLP, CustomTrainState, UNet
from ml4dynamics.trainers import train_utils
from ml4dynamics.types import PRNGKey
from ml4dynamics.utils import calc_utils, viz_utils

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)


def load_data(
  config_dict: ml_collections.ConfigDict,
  batch_size: int,
  num_workers: int = 16,
):

  config = Box(config_dict)
  pde = config.case
  case_num = config.sim.case_num
  if pde == "ns_channel":
    Re = config.sim.Re
    nx = config.sim.nx
    BC = config.sim.BC
    dataset = f"{BC}_Re{Re}_nx{nx}_n{case_num}"
  else:
    sgs = config.train.sgs
    if pde == "react_diff":
      alpha = config.sim.alpha
      beta = config.sim.beta
      gamma = config.sim.gamma
      dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
        alpha, beta, gamma, case_num
      )
    elif pde == "ns_hit":
      Re = config.sim.Re
      dataset = f"Re{Re}_n{case_num}"
    elif pde == "ks":
      bc = "pbc" if config.sim.BC == "periodic" else "dnbc"
      nu = config.sim.nu
      c = config.sim.c
      dataset = f"{bc}_nu{nu:.1f}_c{c:.1f}_n{case_num}"
  h5_filename = f"data/{pde}/{dataset}.h5"

  with h5py.File(h5_filename, "r") as h5f:
    inputs = h5f["data"]["inputs"][()]
    if pde == "ns_channel":
      outputs = h5f["data"]["outputs"][()]
    else:
      outputs = h5f["data"][f"outputs_{sgs}"][()]

  if config.train.input != "global":
    _, model = create_fine_coarse_simulator(config_dict)
    inputs = np.array(augment_inputs(inputs, pde, config.train.input, model))
  # if mode == "torch":
  #   inputs = inputs.transpose(0, 3, 1, 2)
  #   outputs = outputs.transpose(0, 3, 1, 2)
  #   GPU = 0
  #   device = torch.device(
  #     "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
  #   )
  #   inputs = torch.from_numpy(inputs).to(device)
  #   outputs = torch.from_numpy(outputs).to(device)
  if pde == "ks" and config.sim.BC == "Dirichlet-Neumann" and\
    config.train.input == "global":
    # NOTE: the zero-padding is only physically correct for the u and u_x
    inputs_padding = np.zeros((inputs.shape[0], 1, inputs.shape[-1]))
    outputs_padding = np.zeros((outputs.shape[0], 1, outputs.shape[-1]))
    inputs = np.concatenate([inputs, inputs_padding], axis=1)
    outputs = np.concatenate([outputs, outputs_padding], axis=1)
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  train_dataset = TensorDataset(
    torch.tensor(train_x, dtype=torch.float64),
    torch.tensor(train_y, dtype=torch.float64)
  )
  train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # , num_workers=num_workers
  )
  test_dataset = TensorDataset(
    torch.tensor(test_x, dtype=torch.float64),
    torch.tensor(test_y, dtype=torch.float64)
  )
  test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True  # , num_workers=num_workers
  )
  print("inputs shape:", inputs.shape)
  return inputs, outputs, train_dataloader, test_dataloader, dataset


def augment_inputs(inputs: jnp.ndarray, pde: str, input_labels, model):
  """This function is used both at loading the data and inference time"""

  if isinstance(input_labels, int):
    tmp = [inputs]
    if input_labels % 2 != 1:
      raise Exception("Size of stencils must be odd")
    for i in range(input_labels // 2):
      if inputs.ndim == 3:
        tmp.append(jnp.roll(inputs, i + 1, axis=1))
        tmp[-1] = tmp[-1].at[:, :i + 1].set(0)
        tmp.append(jnp.roll(inputs, -(i + 1), axis=1))
        tmp[-1] = tmp[-1].at[:, -i - 1:].set(0)
      elif inputs.ndim == 4:
        tmp.append(jnp.roll(inputs, i + 1, axis=2))
        tmp[-1] = tmp[-1].at[:, :, :i + 1].set(0)
        tmp.append(jnp.roll(inputs, -(i + 1), axis=2))
        tmp[-1] = tmp[-1].at[:, :, -i - 1:].set(0)
  else:
    tmp = []
    if pde == "ks":
      if "u" in input_labels:
        tmp.append(inputs)
      else:
        raise Exception("u is not in input_labels")
      if "u_x" in input_labels:
        if inputs.ndim == 3:
          tmp.append(jnp.einsum("ij, ajk -> aik", model.L1, inputs))
        elif inputs.ndim == 4:
          tmp.append(jnp.einsum("ij, abjk -> abik", model.L1, inputs))
      if "u_xx" in input_labels:
        if inputs.ndim == 3:
          tmp.append(jnp.einsum("ij, ajk -> aik", model.L2, inputs))
        elif inputs.ndim == 4:
          tmp.append(jnp.einsum("ij, abjk -> abik", model.L2, inputs))
      if "u_xxxx" in input_labels:
        if inputs.ndim == 3:
          tmp.append(jnp.einsum("ij, ajk -> aik", model.L4, inputs))
        elif inputs.ndim == 4:
          tmp.append(jnp.einsum("ij, abjk -> abik", model.L4, inputs))
      if "x" in input_labels: 
        if inputs.ndim == 3:
          tmp.append(inputs[..., -1:] if inputs.shape[-1] > 1 else inputs)
        elif inputs.ndim == 4:
          tmp.append(inputs[..., -1:] if inputs.shape[-1] > 1 else inputs)
    elif pde == "ns_hit":
      if "u" in input_labels:
        tmp.append(inputs)
      if "u_x" in input_labels:
        what = jnp.fft.rfft2(inputs, axes=(1, 2))
        psi_hat = -what[..., 0] / model.laplacian_[None]
        dpsidx = jnp.fft.irfft2(1j * psi_hat * model.kx[None], axes=(1, 2))
        dpsidy = jnp.fft.irfft2(1j * psi_hat * model.ky[None], axes=(1, 2))
        tmp.append(dpsidy)
        tmp.append(-dpsidx)

  return jnp.concatenate(tmp, axis=-1)

def create_fine_coarse_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  rng = random.PRNGKey(config.sim.seed)
  T = config.sim.T
  dt = config.sim.dt
  rx = config.sim.rx
  rt = config.sim.rt
  L = config.sim.L
  n = config.sim.n
  # model parameters
  if config.case == "react_diff":
    d = config.sim.d
    model_fine = dynamics.react_diff(
      L=L,
      N=n**2 * 2,
      T=T,
      dt=dt,
      alpha=config.sim.alpha,
      beta=config.sim.beta,
      gamma=config.sim.gamma,
      d=d,
    )
    model_coarse = dynamics.react_diff(
      L=L,
      N=(n // rx)**2 * 2,
      T=T,
      dt=dt * rt,
      alpha=config.sim.alpha,
      beta=config.sim.beta,
      gamma=config.sim.gamma,
      d=d,
    )
  elif config.case == "ks":
    c = config.sim.c
    init_scale = config.sim.init_scale
    BC = config.sim.BC
    # solver parameters
    if BC == "periodic":
      N1 = n
    elif BC == "Dirichlet-Neumann":
      N1 = n - 1
    N2 = N1 // rx
    # fine simulation
    model_fine = dynamics.KS(
      L=L,
      N=N1,
      T=T,
      dt=dt,
      nu=config.sim.nu,
      c=c,
      BC=BC,
      init_scale=init_scale,
      rng=rng,
    )
    # coarse simulator
    model_coarse = dynamics.KS(
      L=L,
      N=N2,
      T=T,
      dt=dt * rt,
      nu=config.sim.nu,
      c=c,
      BC=BC,
      init_scale=init_scale,
      rng=rng,
    )
  elif config.case == "ns_hit":
    model_fine = dynamics.ns_hit(
      L=L * np.pi,
      N=n,
      T=T,
      dt=dt,
      nu=1 / config.sim.Re,
      init_scale=n**(1.5),
    )
    model_coarse = dynamics.ns_hit(
      L=L * np.pi,
      N=n // rx,
      T=T,
      dt=dt * rt,
      nu=1 / config.sim.Re,
      init_scale=(n // rx)**(1.5),
    )
  return model_fine, model_coarse


def create_ns_channel_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  model = dynamics.ns_channel(
    Lx=config.sim.Lx,
    nx=config.sim.nx,
    ny=config.sim.ny,
    T=config.sim.T,
    dt=config.sim.dt,
    Re=config.sim.Re,
  )
  return model


def create_ns_hit_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  model = dynamics.ns_hit(
    L=config.sim.L * np.pi,
    N=config.sim.n,
    T=config.sim.T,
    dt=config.sim.dt,
    nu=1 / config.sim.Re,
    init_scale=config.sim.n**(1.5)
  )
  return model


def hit_init_cond(type_: str, model, key: PRNGKey):

  if type_ == "real_random":
    """initialization scheme 1: random real-space initialization
    simulatoin from this initialization does not have PSD of -5/3 law

    1. the velocity decays fast and the stress become ignorable
    2. the discrepancy between the fine and coarse simulation is very big
    while their energy spectra are similar
    """
    w = random.normal(key, (model.N, model.N))
    w_hat = jnp.fft.rfft2(w)
    w_hat = w_hat.at[0, 0].set(0)
  elif type_ == "spec_random":
    """initialization scheme 2: random spectral-space initialization
    simulatoin from this initialization follows PSD -5/3 law

    the stress is a bit hard to learn
    """
    f0 = 16
    w_hat = jnp.zeros((model.N, model.N // 2 + 1))
    w_hat = w_hat.at[:f0, :f0].set(
      random.normal(key, (f0, f0)) * model.init_scale
    )
    # w_hat = w_hat.at[:f0, :f0].set(model.init_scale)
    w_hat = w_hat.at[0, 0].set(0)
  elif type_ == "gaussian_process":
    r"""initialization scheme 3: Gaussian-process initialization
    w_0 \sim N(0, 7^(2/3)(-\Delta + 49I)^{-2.5})
    reference: https://arxiv.org/pdf/2010.08895
    
    the spectra looks okay
    """
    w_hat = random.normal(
      key, (model.N, model.N // 2 + 1)
    ) * model.init_scale / (-model.laplacian_ + 49)**2.5 * model.N**1.5
  elif type_ == "taylor_green":
    """initialization scheme 4: Taylor-Green vortex
    the stress is zero
    """
    w_hat = jnp.zeros((model.N, model.N // 2 + 1))
    w_hat = w_hat.at[1, 1].set(model.N**2 / 2)
    w_hat = w_hat.at[-1, 1].set(model.N**2 / 2)
  else:
    raise ValueError("Unknown initialization scheme")
  return w_hat


def prepare_unet_train_state(
  config_dict: ml_collections.ConfigDict,
  load_dict: str = None,
  is_global: bool = True,
  is_training: bool = True
):

  config = Box(config_dict)
  rng = random.PRNGKey(config.sim.seed)
  n_sample = int(config.sim.T / config.sim.dt * 0.8)
  inputs_labels = config.train.input
  if inputs_labels == "global":
    epochs = config.train.epochs_global
    decay = config.train.decay_global
    batch_size = config.train.batch_size_global
  else:
    epochs = config.train.epochs_local
    decay = config.train.decay_local
    batch_size = config.train.batch_size_local
  if config.case == "ns_channel":
    nx = config.sim.nx
    ny = config.sim.ny
    input_features = 2
    output_features = 1
    DIM = 2
  else:
    nx = ny = config.sim.n // config.sim.rx
    if config.case == "react_diff":
      input_features = 2
      output_features = 2
      DIM = 2
    elif config.case == "ns_hit":
      if is_global:
        input_features = 1
      else:
        input_features = inputs_labels if isinstance(inputs_labels, int)\
          else len(inputs_labels)
      output_features = 1
      DIM = 2
    elif config.case == "ks":
      if is_global:
        input_features = 1
      else:
        input_features = inputs_labels if isinstance(inputs_labels, int)\
          else len(inputs_labels)
      output_features = 1
      DIM = 1
  if load_dict:
    with open(f"{load_dict}", "rb") as f:
      params = pickle.load(f)
  step_per_epoch = n_sample * config.sim.case_num // batch_size + 1
  schedule = optax.piecewise_constant_schedule(
    init_value=config.train.lr,
    boundaries_and_scales={
      int(b): 0.1
      for b in jnp.arange(
        decay * step_per_epoch, epochs * step_per_epoch, decay * step_per_epoch
      )
    }
  )
  optimizer = optax.adam(schedule)
  if is_global:
    unet = UNet(
      input_features=input_features,
      output_features=output_features,
      kernel_size=config.train.kernel_size,
      DIM=DIM,
      training=is_training,
    )
    rng1, rng2 = random.split(rng)
    init_rngs = {'params': rng1, 'dropout': rng2}
    if not load_dict:
      if config.case == "ks":
        # init 1D UNet
        params = unet.init(
          init_rngs, jnp.ones([1, nx, input_features], dtype=jnp.float64)
        )
      else:
        params = unet.init(
          init_rngs, jnp.ones([1, nx, ny, input_features], dtype=jnp.float64)
        )
    train_state = CustomTrainState.create(
      apply_fn=unet.apply,
      params=params["params"],
      tx=optimizer,
      batch_stats=params["batch_stats"]
    )
  else:
    mlp = MLP(output_features)
    if not load_dict:
      params = mlp.init(
        random.PRNGKey(0), np.zeros((1, input_features), dtype=jnp.float64)
      )
    train_state = TrainState.create(
      apply_fn=mlp.apply,
      params=params,
      tx=optimizer,
    )

  flat_params = traverse_util.flatten_dict(train_state.params)
  for _, value in flat_params.items():
    if isinstance(value, jnp.ndarray) and value.dtype != jnp.float64:
      breakpoint()
      raise Exception(
        f"Parameter {value} is not in float64, please check the model"
      )
  return train_state, schedule


def data_stratification(config_dict: ml_collections.ConfigDict):
  config = Box(config_dict)
  # stratify the data to balance it
  if config.train.stratify == "input_output":
    bins = config.train.bins
    subsample = config.train.subsample
    hist, xedges, yedges = np.histogram2d(
      inputs.reshape(-1), outputs.reshape(-1), bins=bins
    )
    bin_data = {}
    for i in range(bins):
      for j in range(bins):
        if hist[i, j] > 0:
          bin_mask = (xedges[i] <= inputs) & (inputs < xedges[i + 1]) &\
            (yedges[j] <= outputs) & (outputs < yedges[j + 1])
          bin_pts = np.column_stack((inputs[bin_mask], outputs[bin_mask]))
          bin_data[(i, j)] = bin_pts
    stratify_inputs = jnp.zeros((1, 1))
    stratify_outputs = jnp.zeros((1, 1))
    for _ in bin_data.keys():
      if bin_data[_].shape[0] < subsample:
        stratify_inputs = jnp.vstack([stratify_inputs, bin_data[_][:, 0:1]])
        stratify_outputs = jnp.vstack([stratify_outputs, bin_data[_][:, 1:]])
      else:
        samples = np.random.choice(
          jnp.arange(bin_data[_].shape[0]), size=subsample, replace=False
        )
        stratify_inputs = jnp.vstack(
          [stratify_inputs, bin_data[_][samples, 0:1]]
        )
        stratify_outputs = jnp.vstack(
          [stratify_outputs, bin_data[_][samples, 1:]]
        )
    inputs = stratify_inputs
    outputs = stratify_outputs

  if config.train.stratify == "input":
    bins = config.train.bins
    subsample = config.train.subsample
    hist, xedges, yedges = np.histogram2d(
      inputs.reshape(-1), outputs.reshape(-1), bins=bins
    )
    bin_data = {}
    for i in range(bins):
      for j in range(bins):
        if hist[i, j] > 0:
          bin_mask = (xedges[i] <= inputs) & (inputs < xedges[i + 1]) &\
            (yedges[j] <= outputs) & (outputs < yedges[j + 1])
          bin_pts = np.column_stack((inputs[bin_mask], outputs[bin_mask]))
          bin_data[(i, j)] = bin_pts
    stratify_inputs = jnp.zeros((1, 1))
    stratify_outputs = jnp.zeros((1, 1))
    for _ in bin_data.keys():
      if bin_data[_].shape[0] < subsample:
        stratify_inputs = jnp.vstack([stratify_inputs, bin_data[_][:, 0:1]])
        stratify_outputs = jnp.vstack([stratify_outputs, bin_data[_][:, 1:]])
      else:
        samples = np.random.choice(
          jnp.arange(bin_data[_].shape[0]), size=subsample, replace=False
        )
        stratify_inputs = jnp.vstack(
          [stratify_inputs, bin_data[_][samples, 0:1]]
        )
        stratify_outputs = jnp.vstack(
          [stratify_outputs, bin_data[_][samples, 1:]]
        )
    inputs = stratify_inputs
    outputs = stratify_outputs


def eval_a_priori(
  forward_fn: callable,
  train_dataloader: DataLoader,
  test_dataloader: DataLoader,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
  dim: int = 2,
  fig_name: str = None,
):

  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in train_dataloader:
    predict = forward_fn(jnp.array(batch_inputs))
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"train loss: {total_loss/count:.4e}")
  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in test_dataloader:
    predict = forward_fn(jnp.array(batch_inputs))
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"test loss: {total_loss/count:.4e}")

  if dim == 1:
    predicts = forward_fn(jnp.array(inputs))
    im_list = [
      outputs[..., 0].T, predicts[..., 0].T, (outputs - predicts)[..., 0].T
    ]
    plot_with_horizontal_colorbar(
      im_list, None, [len(im_list), 1], None, f"results/fig/{fig_name}.png", 100
    )
  elif dim == 2:
    # visualization
    n_plot = 4
    index_array = np.arange(
      0, n_plot * outputs.shape[0] // n_plot - 1, outputs.shape[0] // n_plot
    )
    im_array = np.zeros((3, n_plot, *(outputs[0, ..., 0]).shape))
    for j in range(n_plot):
      predicts = forward_fn(
        jnp.array(inputs[index_array[j]:index_array[j] + 1])
      )
      im_array[0, j] = outputs[index_array[j], ..., 0]
      im_array[1, j] = predicts[index_array[j], ..., 0]
      im_array[2, j] = (outputs - predicts)[index_array[j], ..., 0]
    im_list = []
    for j in range(im_array.shape[0] * im_array.shape[1]):
      im_list.append(im_array[j // n_plot, j % n_plot])
    plot_with_horizontal_colorbar(
      im_list, None, [3, n_plot], None, f"results/fig/{fig_name}.png", 100
    )

    # im_array = np.zeros((2, n_plot, *(outputs[0, ..., 0].T).shape))
    # title_array1 = []
    # title_array2 = []
    # for j in range(n_plot):
    #   im_array[0, j] = inputs[index_array[j], ..., 0].T
    #   im_array[1, j] = outputs[index_array[j], ..., 0].T
    #   title_array1.append(f"{inputs[index_array[j], ..., 0].max():.1e}")
    #   title_array2.append(f"{outputs[index_array[j], ..., 0].max():.1e}")
    # plot_with_horizontal_colorbar(
    #   im_array, (12, 6), title_array1 + title_array2,
    #   f"results/fig/dataset1.png", 100
    # )
  plt.hist(
    inputs.reshape(-1), bins=100, density=True, alpha=0.3, label="inputs"
  )
  plt.hist(
    outputs.reshape(-1), bins=100, density=True, alpha=0.3, label="outputs"
  )
  plt.yscale("log")
  plt.legend()
  plt.savefig(f"results/fig/dataset.png")
  plt.close()


def eval_a_posteriori(
  config_dict: ml_collections.ConfigDict,
  forward_fn: callable,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
  dim: int = 2,
  beta: float = 0.0,
  fig_name: str = None,
  _plot: bool = True,
):

  config = Box(config_dict)
  if config.case == "ns_channel":
    model = create_ns_channel_simulator(config)
    run_simulation = partial(
      train_utils.run_ns_simulation_pressue_correction, forward_fn, model,
      outputs, beta
    )
  else:
    model_fine, model = create_fine_coarse_simulator(config)
    type_ = None
    if config.case == "react_diff":
      iter_ = model.adi
    elif config.case == "ns_hit":
      iter_ = model.CN_real
    elif config.case == "ks":
      iter_ = model.CN_FEM
      if config.sim.BC == "Dirichlet-Neumann":
        type_ = "pad"
    if config.train.sgs == "fine_correction":
      res_fn, int_fn = res_int_fn(config)
      run_simulation = partial(
        train_utils.run_simulation_fine_grid_correction, forward_fn, model,
        iter_, outputs, beta, res_fn, int_fn, type_
      )
    elif config.train.sgs == "filter":
      run_simulation = partial(
        train_utils.run_simulation_sgs, forward_fn, model, iter_, outputs, beta,
        type_
      )
    elif config.train.sgs == "correction":
      run_simulation = partial(
        train_utils.run_simulation_coarse_grid_correction, forward_fn, model,
        iter_, outputs, beta, type_
      )

  step_num = model.step_num
  t_array = np.linspace(0, model.T, model.step_num)
  stats_test = True
  if not stats_test:
    start = time()
    inputs = inputs[:step_num]
    outputs = outputs[:step_num]
    if config.case == "ns_hit":
      model.set_x_hist(np.fft.rfft2(inputs[0, ..., 0]), model.CN)
    elif config.case == "ks":
      if inputs.ndim == 3:
        model.run_simulation(inputs[0, :, 0], iter_)
      elif inputs.ndim == 4:
        model.run_simulation(inputs[0, :, 0, 0], iter_)
    x_hist = run_simulation(inputs[0])
    print(f"simulation takes {time() - start:.2f}s...")
    if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
      print("similation contains NaN!")
    print(
      "baseline:",
      jnp.mean(jnp.linalg.norm(inputs - model.x_hist[..., None], axis=(1, 2)))
    )
    print("ours:", jnp.mean(jnp.linalg.norm(inputs - x_hist, axis=(1, 2))))
    x_hist = jnp.where(jnp.abs(x_hist) < 5, x_hist, 5)
    model.x_hist = jnp.where(jnp.abs(model.x_hist) < 5, model.x_hist, 5)

    if dim == 1 and _plot:

      im_list = [
        inputs[..., 0].T, model.x_hist.T, (inputs[..., 0] - model.x_hist).T,
        x_hist[..., 0].T, (inputs - x_hist)[..., 0].T
      ]
      plot_with_horizontal_colorbar(
        im_list, None, [len(im_list), 1], None, f"results/fig/{fig_name}.png",
        100
      )
      viz_utils.plot_stats_aux(
        np.arange(inputs.shape[0]) * model.dt,
        [inputs[..., 0], model.x_hist, x_hist[..., 0]],
        ["truth", "baseline", "ours"],
        f"results/fig/{fig_name}_stats.png",
      )
      viz_utils.plot_temporal_corr(
        [inputs[..., 0], model.x_hist, x_hist[..., 0]], ['baseline', 'ours'],
        t_array, fig_name
      )
    elif dim == 2 and _plot:
      n_plot = 6
      step_num = inputs.shape[0]
      index_array = np.arange(
        0, n_plot * step_num // n_plot - 1, step_num // n_plot
      )
      im_array = np.zeros((5, n_plot, *(outputs[0, ..., 0]).shape))
      for j in range(n_plot):
        im_array[0, j] = inputs[index_array[j], ..., 0]
        im_array[1, j] = model.x_hist[index_array[j]]
        im_array[3, j] = x_hist[index_array[j], ..., 0]
        im_array[2, j] = inputs[index_array[j], ..., 0] -\
          model.x_hist[index_array[j]]
        im_array[4, j] = inputs[index_array[j], ..., 0] -\
          x_hist[index_array[j], ..., 0]
      im_list = []
      for j in range(im_array.shape[0] * im_array.shape[1]):
        im_list.append(im_array[j // n_plot, j % n_plot])
      plot_with_horizontal_colorbar(
        im_list, None, [5, n_plot], None, f"results/fig/{fig_name}.png", 100
      )

      if config.case == "ns_hit":
        u_hist_true = np.zeros((*x_hist.shape[:-1], 2))
        u_hist = np.zeros((*x_hist.shape[:-1], 2))
        u_hist_baseline = np.zeros((*x_hist.shape[:-1], 2))
        psi_true = jnp.fft.rfft2(inputs[..., 0],
                                 axes=(1, 2)) / model.laplacian_[None]
        psi = jnp.fft.rfft2(x_hist[..., 0],
                            axes=(1, 2)) / model.laplacian_[None]
        psi_baseline = jnp.fft.rfft2(model.x_hist, axes=(1, 2)) /\
          model.laplacian_[None]
        u_hist_true[
          ...,
          1] = -jnp.fft.irfft2(1j * psi_true * model.kx[None], axes=(1, 2))
        u_hist_true[
          ..., 0] = jnp.fft.irfft2(1j * psi_true * model.ky[None], axes=(1, 2))
        u_hist[..., 1] = -jnp.fft.irfft2(1j * psi * model.kx[None], axes=(1, 2))
        u_hist[..., 0] = jnp.fft.irfft2(1j * psi * model.ky[None], axes=(1, 2))
        u_hist_baseline[
          ...,
          1] = -jnp.fft.irfft2(1j * psi_baseline * model.kx[None], axes=(1, 2))
        u_hist_baseline[
          ...,
          0] = jnp.fft.irfft2(1j * psi_baseline * model.ky[None], axes=(1, 2))
        viz_utils.plot_psd_cmp(
          [u_hist_true, u_hist_baseline, u_hist],
          [[model.dx, model.dx], [model.dx, model.dx], [model.dx, model.dx]],
          ["truth", "baseline", "ours"], fig_name
        )
  else:
    n_sample = 10
    rng = random.PRNGKey(1000)
    res_fn, _ = res_int_fn(config_dict)
    L = model_fine.Lx
    N1 = config.sim.n
    dx = L / N1
    l2_list = []
    first_moment_list = []
    second_moment_list = []
    corr1 = []
    corr2 = []
    avg_length = 1000
    for _ in range(n_sample):
      print(f"{_}th sample")
      rng, key = random.split(rng)
      r0 = random.uniform(key) * 20 + 44
      if config.sim.BC == "periodic":
        r0 = random.uniform(key) * 20 + 44
        u0 = jnp.exp(-(jnp.linspace(0, L - L / N1, N1) - r0)**2 / r0**2 * 4)
      elif config.sim.BC == "Dirichlet-Neumann":
        x = jnp.linspace(dx, L - dx, N1 - 1)
        r0 = random.uniform(key) * 20 + 44
        u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
      model_fine.run_simulation(u0, model_fine.CN_FEM)
      truth = jax.vmap(res_fn)(model_fine.x_hist)[..., 0]
      model.run_simulation(res_fn(u0)[..., 0], iter_)
      x_hist = run_simulation(res_fn(u0))
      if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
        print("similation contains NaN!")
      baseline_l2 = jnp.mean(jnp.linalg.norm(truth - model.x_hist, axis=(1, )))
      ours_l2 = jnp.mean(jnp.linalg.norm(truth - x_hist[..., 0], axis=(1, )))
      baseline_first_moment = np.mean((truth - model.x_hist)[-avg_length:])
      ours_first_moment = np.mean((truth - x_hist[..., 0])[-avg_length:])
      baseline_second_moment = np.mean(
        (truth**2 - model.x_hist**2)[-avg_length:]
      )
      ours_second_moment = np.mean((truth**2 - x_hist[..., 0]**2)[-avg_length:])
      l2_list.append(np.array([baseline_l2, ours_l2]))
      first_moment_list.append(
        np.array([baseline_first_moment, ours_first_moment])
      )
      second_moment_list.append(
        np.array([baseline_second_moment, ours_second_moment])
      )
      print("l2:", l2_list[-1])
      print("first moment:", first_moment_list[-1])
      print("second moment:", second_moment_list[-1])

      corr1.append(calc_utils.calc_temporal_corr(truth, model.x_hist))
      corr2.append(calc_utils.calc_temporal_corr(truth, x_hist[..., 0]))

    corr1 = np.array(corr1)
    corr2 = np.array(corr2)
    plt.plot(t_array, np.mean(corr1, axis=0), label="baseline")
    plt.plot(t_array, np.mean(corr2, axis=0), label="ours")
    plt.legend()
    plt.savefig(f"results/fig/{fig_name}_corr.png")
    breakpoint()
    l2_list = np.array(l2_list)
    first_moment_list = np.array(first_moment_list)
    second_moment_list = np.array(second_moment_list)
    print(
      np.mean(l2_list[~np.isnan(l2_list).any(axis=1)], axis=0), ", ",
      np.std(l2_list[~np.isnan(l2_list).any(axis=1)], axis=0)
    )
    print(
      np.mean(
        first_moment_list[~np.isnan(first_moment_list).any(axis=1)], axis=0
      ), ", ",
      np.std(
        first_moment_list[~np.isnan(first_moment_list).any(axis=1)], axis=0
      )
    )
    print(
      np.mean(
        second_moment_list[~np.isnan(second_moment_list).any(axis=1)], axis=0
      ), ", ",
      np.std(
        second_moment_list[~np.isnan(second_moment_list).any(axis=1)], axis=0
      )
    )
  viz_utils.plot_stats_aux(
    np.arange(inputs.shape[0]) * model.dt,
    [inputs[..., 0], model.x_hist, x_hist[..., 0]],
    ["truth", "baseline", "ours"],
    f"results/fig/{fig_name}_stats.png",
  )
  breakpoint()
  return x_hist


###############################################################################
#                   Numerical solver of the reaction-diffusion equation:
# For linear term, we use different discretization scheme, e.g. explicit,
#  implicit, Crank-Nielson, ADI etc.
# For nonlinear term, we use the explicit scheme (need to implement Ashford)
###############################################################################


def assembly_RDmatrix(n, dt, dx, beta=1.0, gamma=0.05, d=2):
  r"""assemble matrices used in the calculation

    A1 = I - \gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - \gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + \gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization

    :d: ratio between the diffusion coeff for u & v
    """

  global L_uminus, L_uplus, L_vminus, L_vplus, A_uplus, A_uminus, A_vplus, A_vminus, \
            A0_u, A0_v, L, L2

  L = jnp.eye(n) * -2 + jnp.eye(n, k=1) + jnp.eye(n, k=-1)
  L = L.at[0, -1].set(1)
  L = L.at[-1, 0].set(1)
  L = L / (dx**2)
  L2 = jnp.kron(L, jnp.eye(n)) + jnp.kron(jnp.eye(n), L)

  # matrix for ADI scheme
  L_uminus = jnp.eye(n) - L * gamma * dt / 2
  L_uplus = jnp.eye(n) + L * gamma * dt / 2
  L_vminus = jnp.eye(n) - L * gamma * dt / 2 * d
  L_vplus = jnp.eye(n) + L * gamma * dt / 2 * d

  A0_u = jnp.eye(n * n) + L2 * gamma * dt
  A0_v = jnp.eye(n * n) + L2 * gamma * dt * d
  A_uplus = jnp.eye(n * n) + L2 * gamma * dt / 2
  A_uminus = jnp.eye(n * n) - L2 * gamma * dt / 2
  A_vplus = jnp.eye(n * n) + L2 * gamma * dt / 2 * d
  A_vminus = jnp.eye(n * n) - L2 * gamma * dt / 2 * d
  # L = spa.csc_matrix(L)
  # L_uminus = spa.csc_matrix(L_uminus)
  # L_uplus = spa.csc_matrix(L_uplus)
  # L_vminus = spa.csc_matrix(L_vminus)
  # L_vplus = spa.csc_matrix(L_vplus)
  # L2 = spa.kron(L, np.eye(n)) + spa.kron(np.eye(n), L)
  # A_uplus = spa.eye(n*n) + L2 * gamma * dt/2
  # A_uminus = spa.eye(n*n) - L2 * gamma * dt/2
  # A_vplus = spa.eye(n*n) + L2 * gamma * dt/2 * d
  # A_vminus = spa.eye(n*n) - L2 * gamma * dt/2 * d


def RD_exp(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """
    explicit forward Euler solver for FitzHugh-Nagumo RD equation
    :input:
    u, v: initial condition, shape [nx, ny], different nx, ny is used for
    different diffusion coeff
    """

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    tmpu = A0_u @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    tmpv = A0_v @ v + beta * dt * (u - v) + rhsv_ * dt
    u = tmpu
    v = tmpv

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist


def RD_semi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """semi-implicit solver for FitzHugh-Nagumo RD equation"""

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    rhsu = A_uplus @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    rhsv = A_vplus @ v + beta * dt * (u - v) + rhsv_ * dt
    u, _ = jsla.cg(A_uminus, rhsu)
    v, _ = jsla.cg(A_vminus, rhsv)

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist


def RD_adi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """ADI solver for FitzHugh-Nagumo RD equation"""

  u = jnp.array(u)
  v = jnp.array(v)

  @jax.jit
  def update(u, v):

    rhsu = rhsu_ * dt + L_uplus @ u @ L_uplus + dt * (u - v - u**3 + alpha)
    rhsv = rhsv_ * dt + L_vplus @ v @ L_vplus + beta * dt * (u - v)
    return solve(L_uminus, solve(L_uminus, rhsu).T).T,\
      solve(L_uminus, solve(L_vminus, rhsv).T).T

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx, ny)
    rhsv_ = source[1].reshape(nx, ny)
  flag = True

  for i in range(step_num):
    u, v = update(u, v)
    if jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)) or jnp.any(
      jnp.isinf(u)
    ) or jnp.any(jnp.isinf(v)):
      flag = False
      break

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval, :].set(u)
      v_hist = v_hist.at[(i - 0) // writeInterval, :].set(v)

  return u_hist, v_hist, flag


def RD_cn(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """full implicit solver with Crank-Nielson discretization
    TODO: we have not tested this function yet"""

  global L, D_
  dt = 1 / step_num
  t_array = np.array([5, 10, 20, 40, 80])

  #t_array = np.array([1, 2, 3, 4, 80])

  #plt.subplot(231)
  #plt.imshow(u.reshape(n, n), cmap = cm.viridis)

  def F(u_next, v_next, u, v):
    Fu = A2 @ u_next - A3 @ u + (
      u_next**3 + u**3 + v_next + v - u_next - u - alpha
    ) * dt / 2
    Fv = A2 @ v_next - A3 @ v + (v_next + v - u_next - u) * dt * beta / 2
    res = np.hstack([Fu, Fv])
    return res

  def Newton(n):

    global u, v, L, D_
    # we use the semi-implicit scheme iteration as the initial guess of Newton method
    rhsu = u + dt * (u - v + u**3 + alpha)
    rhsv = v + beta * dt * (u - v)
    u_next = sps(A1, rhsu)
    v_next = sps(A1, rhsv)
    res = F(u_next, v_next, u, v)

    count = 0
    while jnp.linalg.norm(res) > tol:
      D_[:n * n, :n *
         n] = D_[:n * n, :n *
                 n] + dt / 2 * (spa.diags(3 * (u_next**2)) - spa.eye(n * n))
      D = D_.tocsr()
      #duv = sps(D, res)
      # GMRES with initial guess pressure in last time step
      duv = jsla.gmres(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      # BiSTABCG with initial guess pressure in last time step
      duv = jsla.bicgstab(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      u_next = u_next - duv[:n * n]
      v_next = v_next - duv[n * n:]
      res = F(u_next, v_next, u, v)
      count = count + 1
      print(scalg.norm(res))
    print(count)

    u = u_next
    v = v_next

  for i in range(step_num):
    for j in range(5):
      #if i == t_array[j] * step_num / 100:
      #    plt.subplot(2, 3, j+2)
      #    plt.imshow(u.reshape(n, n), cmap = cm.viridis)
      #    plt.colorbar()
      Newton()

  #plt.show()
  return u, v


def assembly_NSmatrix(nx, ny, dx, dy, BC: str = "Dirichlet"):
  """assemble matrices used in the calculation
    LD: Laplacian operator with Dirichlet BC
    LN: Laplacian operator with Neuman BC, notice that this operator may have
    different form depends on the position of the boundary, here we use the
    case that boundary is between the outmost two grids
    L:  Laplacian operator associated with current BC with three Neuman BCs on
    upper, lower, left boundary and a Dirichlet BC on right
    """

  global L

  def Laplacian_Neumann(n):
    LN = jnp.roll(jnp.eye(n), 1, axis=1) + jnp.roll(jnp.eye(n), -1, axis=1) -\
      jnp.eye(n) * 2
    LN = LN.at[0, 0].set(-1)
    LN = LN.at[0, -1].set(0)
    LN = LN.at[-1, -1].set(-1)
    LN = LN.at[-1, 0].set(0)
    return LN

  LNx = Laplacian_Neumann(nx)
  LNy = Laplacian_Neumann(ny)
  L = jnp.kron(LNx /
               (dx**2), jnp.eye(ny)) + jnp.kron(jnp.eye(nx), LNy / (dy**2))
  if BC == "Dirichlet":
    for i in range(ny):
      L = L.at[-1 - i, -1 - i].add(-2 / (dx**2))
  elif BC == "Neumann":
    L = jnp.vstack([L, jnp.ones_like(L[0:1])])
    L = jnp.hstack([L, jnp.ones_like(L[:, 0:1])])
    L = L.at[-1, -1].set(0)


def projection_correction(
  u: jnp.ndarray,
  v: jnp.ndarray,
  p: jnp.ndarray,
  t: float,
  dx=1 / 32,
  dy=1 / 32,
  nx=128,
  ny=32,
  y0=0.325,
  dt=.01,
  Re=100,
  BC: str = "Dirichlet",
  correction: bool = False,
):
  r"""projection method to solve the incompressible NS equation
    The convection discretization is given by central difference
    u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx
    
  The collocation point of p locates at the center of the cell (1/2, 1/2)
  The collocation point of u locates at the right of p (1, 1/2)
  The collocation point of v locates at the top of p (1/2, 1)
  v[:, -1] = 0 for the no-slip boundary condition

  """

  def _u_padx(u: jnp.ndarray):
    return jnp.vstack([u_inlet, u, u[-1]])

  def _v_padx(v: jnp.ndarray):
    return jnp.vstack([2 * v_inlet - v[0], v, v[-1]])

  def _u_pady(u: jnp.ndarray):
    return jnp.hstack([-u[:, 0:1], u, -u[:, -1:]])

  def _v_pady(v: jnp.ndarray):
    return jnp.hstack([jnp.zeros_like(v[:, 0:1]), v, -v[:, -2:-1]])

  def _v2u(v: jnp.ndarray):
    """interpolate v to u"""
    v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
    v = jnp.vstack([v_pady, v_pady[-1]])
    return (v[1:, 1:] + v[:-1, 1:] + v[1:, :-1] + v[:-1, :-1]) / 4

  def _u2v(u: jnp.ndarray):
    """interpolate u to v"""
    u_padx = jnp.vstack([u_inlet, u])
    u = jnp.hstack([u_padx, -u_padx[:, -1:]])
    return (u[1:, 1:] + u[:-1, 1:] + u[1:, :-1] + u[:-1, :-1]) / 4

  def grad_p(p: jnp.ndarray, dpdn: float = 0):
    """calculate the gradient of the pressure
    
    p: (nx, ny)
    dpdx: (nx - 1, ny)
    dpdy: (nx, ny - 1)
    dpdx[-1] = dpdy[:, -1] = 0 since p satisfies the Neuman BC 
    """
    if BC == "Dirichlet":
      p_padx = jnp.vstack([p, -p[-1:]])
    elif BC == "Neumann":
      p_padx = jnp.vstack([p, p[-1:] + dpdn * dx])
    dpdx = (p_padx[1:] - p_padx[:-1]) / dx
    dpdy = (p[:, 1:] - p[:, :-1]) / dy
    return dpdx, dpdy

  def div_uv(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the divergence of the velocity field"""
    u_padx = jnp.vstack([u_inlet, u])
    dudx = (u_padx[1:] - u_padx[:-1]) / dx
    v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
    dvdy = (v_pady[:, 1:] - v_pady[:, :-1]) / dy
    return dudx + dvdy

  def laplace_uv(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the Laplacian of the velocity field"""

    u_padx = _u_padx(u)
    u_pad = _u_pady(u_padx)
    lapl_u = (u_pad[2:, 1:-1] - 2 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2 +\
      (u_pad[1:-1, 2:] - 2 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2
    v_padx = _v_padx(v)
    v_pad = _v_pady(v_padx)
    lapl_v = (v_pad[1:-1, 2:] - 2 * v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dy**2 +\
      (v_pad[2:, 1:-1] - 2 * v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dx**2
    return lapl_u, lapl_v

  def transport(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the transport term of the velocity field"""
    u_padx = _u_padx(u)
    u_pady = _u_pady(u)
    uu_x = (u_padx[2:] - u_padx[:-2]) / dx / 2 * u
    vu_y = (u_pady[:, 2:] - u_pady[:, :-2]) / dy / 2 * _v2u(v)
    v_padx = _v_padx(v)
    v_pady = _v_pady(v)
    uv_x = (v_padx[2:] - v_padx[:-2]) / dx / 2 * _u2v(u)
    vv_y = (v_pady[:, 2:] - v_pady[:, :-2]) / dy / 2 * v

    return uu_x + vu_y, uv_x + vv_y

  def inlet(y: jnp.ndarray):
    """set the inlet velocity"""
    return y * (1 - y) * jnp.exp(-10 * (y - y0)**2) * 3

  u_inlet = inlet(np.linspace(dy / 2, 1 - dy / 2, ny))
  v_inlet = inlet(np.linspace(dy, 1, ny)) * jnp.sin(t)

  dpdx, dpdy = grad_p(p)
  lapl_u, lapl_v = laplace_uv(u, v)
  du, dv = transport(u, v)
  u += -dpdx * dt
  v = v.at[:, :-1].add(-dpdy * dt)
  u += dt * (lapl_u / Re - du)
  v += dt * (lapl_v / Re - dv)
  v = v.at[:, -1].set(0)

  if not correction:
    # pressure correction
    res = div_uv(u, v) / dt
    if BC == "Dirichlet":
      p_res = jnp.linalg.solve(L, res.reshape(-1)).reshape([nx, ny])
      dpdn = 0
    elif BC == "Neumann":
      dpdn = -res.sum() / ny * dx
      res = res.at[-1].add(dpdn / dx)
      p_res = jnp.linalg.solve(L, jnp.hstack([res.reshape(-1),
                                              jnp.zeros(1)]
                                             ))[:-1].reshape([nx, ny])

    dpdx, dpdy = grad_p(p_res, -dpdn)
    u += -dpdx * dt
    v = v.at[:, :-1].add(-dpdy * dt)
    p += p_res

  res_ = div_uv(u, v)
  if jnp.linalg.norm(res_) > 1e-12:
    print(jnp.linalg.norm(res_))
    print("Velocity field is not divergence free!!!")

  return u, v, p


def a_posteriori_analysis(
  config: ml_collections.ConfigDict,
  ks_fine,
  ks_coarse,
  correction_nn: callable,
  params,
):

  c = config.sim.c
  L = config.sim.L
  T = config.sim.T
  init_scale = config.sim.init_scale
  BC = config.sim.BC
  # solver parameters
  if BC == "periodic":
    N1 = config.sim.n
  elif BC == "Dirichlet-Neumann":
    N1 = config.sim.n - 1
  r = config.sim.rx
  N2 = N1 // r
  rng = random.PRNGKey(config.sim.seed)
  train_mode = config.train.mode
  n_g = config.train.n_g

  # a posteriori analysis
  rng, key = random.split(rng)
  if BC == "periodic":
    u0 = ks_fine.attractor + init_scale * random.normal(key) *\
      jnp.sin(10 * jnp.pi * jnp.linspace(0, L - L/N1, N1) / L)
  elif BC == "Dirichlet-Neumann":
    dx = L / (N1 + 1)
    x = jnp.linspace(dx, L - dx, N1)
    # u0 = ks_fine.attractor + init_scale * random.normal(key) *\
    #   jnp.sin(10 * jnp.pi * x / L)
    # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
    #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
    r0 = random.uniform(key) * 20 + 44
    u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
  ks_fine.run_simulation(u0, ks_fine.CN_FEM)
  if config.test.solver == "CN":
    ks_coarse.run_simulation(u0[r - 1::r], ks_coarse.CN_FEM)
  elif config.test.solver == "RK4":
    ks_coarse.run_simulation(u0[r - 1::r], ks_coarse.RK4)
  # im_array = jnp.zeros(
  #   (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  # )
  # im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, r-1::r].T)
  # im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  # im_array = im_array.at[2,
  #                        0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, r-1::r].T)
  # title_array = [f"{N1}", f"{N2}", "diff"]
  baseline = ks_coarse.x_hist

  if train_mode == "regression":

    def corrector(input):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      if config.train.input == "u":
        return partial(correction_nn.apply,
                       params)(input.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "ux":
        return partial(correction_nn.apply, params)(u_x.reshape(-1,
                                                                1)).reshape(-1)
      elif config.train.input == "uxx":
        return partial(correction_nn.apply,
                       params)(u_xx.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "uxxxx":
        return partial(correction_nn.apply,
                       params)(u_xxxx.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "global":
        return partial(correction_nn.apply,
                       params)(input.reshape(1, -1)).reshape(-1)
      # global model: [N] to [N]
      # return partial(correction_nn.apply,
      #                params)(input.reshape(1, -1)).reshape(-1)

  elif train_mode == "generative":
    vae_bind = vae.bind({"params": params})
    z = random.normal(key, shape=(1, config.train.vae.latents))
    corrector = partial(vae_bind.generate, z)
  elif train_mode == "gaussian":
    z = random.normal(key)

    def corrector(input: jnp.array):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      if config.train.input == "u":
        tmp = partial(correction_nn.apply, params)(input.reshape(-1, 1))
      elif config.train.input == "ux":
        tmp = partial(correction_nn.apply, params)(u_x.reshape(-1, 1))
      elif config.train.input == "uxx":
        tmp = partial(correction_nn.apply, params)(u_xx.reshape(-1, 1))
      elif config.train.input == "uxxxx":
        tmp = partial(correction_nn.apply, params)(u_xxxx.reshape(-1, 1))
      p = jax.nn.sigmoid(tmp[..., :n_g])
      index = jax.vmap(partial(jax.random.choice, key=key, a=n_g,
                               shape=(1, )))(p=p).reshape(-1)
      return (
        tmp[np.arange(N2), n_g + index] + tmp[np.arange(N2), n_g + index] * z
      ).reshape(-1)

    def corrector_sample(input: jnp.array, rng: PRNGKey):

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      if config.train.input == "u":
        tmp = partial(correction_nn.apply, params)(input.reshape(-1, 1))
      elif config.train.input == "ux":
        tmp = partial(correction_nn.apply, params)(u_x.reshape(-1, 1))
      elif config.train.input == "uxx":
        tmp = partial(correction_nn.apply, params)(u_xx.reshape(-1, 1))
      elif config.train.input == "uxxxx":
        tmp = partial(correction_nn.apply, params)(u_xxxx.reshape(-1, 1))
      p = jax.nn.sigmoid(tmp[..., :n_g])
      index = jax.vmap(partial(jax.random.choice, key=key, a=n_g,
                               shape=(1, )))(p=p).reshape(-1)
      z = random.normal(key)
      return (
        tmp[np.arange(N2), n_g + index] + tmp[np.arange(N2), n_g + index] * z
      ).reshape(-1)

  if config.test.solver == "CN":
    ks_coarse.run_simulation_with_correction(
      u0[r - 1::r], ks_coarse.CN_FEM, corrector
    )
  elif config.test.solver == "RK4":
    ks_coarse.run_simulation_with_correction(
      u0[r - 1::r], ks_coarse.RK4, corrector
    )
  correction1 = ks_coarse.x_hist
  correction2 = None
  if train_mode == "gaussian":
    if config.test.solver == "CN":
      ks_coarse.run_simulation_with_probabilistic_correction(
        u0[r - 1::r], ks_coarse.CN_FEM, corrector_sample
      )
    elif config.test.solver == "RK4":
      ks_coarse.run_simulation_with_probabilistic_correction(
        u0[r - 1::r], ks_coarse.RK4, corrector_sample
      )
    correction2 = ks_coarse.x_hist

  # compare the simulation statistics (A posteriori analysis)
  viz_utils.plot_stats(
    np.arange(ks_fine.x_hist.shape[0]) * ks_fine.dt,
    ks_fine.x_hist,
    baseline,
    correction1,
    correction2,
    f"results/fig/ks_c{c}T{T}n{config.sim.case_num}_{train_mode}_stats.png",
  )
  # plot_with_horizontal_colorbar(
  #   im_array,
  #   fig_size=(4, 6),
  #   title_array=title_array,
  #   file_path=
  #   f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_cmp.pdf"
  # )
  # im_array = jnp.zeros(
  #   (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  # )
  # im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, r-1::r].T)
  # im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  # im_array = im_array.at[2,
  #                        0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, r-1::r].T)
  # plot_with_horizontal_colorbar(
  #   im_array,
  #   fig_size=(4, 6),
  #   title_array=title_array,
  #   file_path=
  #   f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_correct_cmp.pdf"
  # )


def plot_with_horizontal_colorbar(
  im_list: list,
  title_array: list = None,
  shape: list = None,
  fig_size: tuple = None,
  file_path: list = None,
  dpi: int = 100,
  cmap=cm.twilight
):

  if np.prod(shape) != len(im_list):
    breakpoint()
    raise Exception("the number of images does not match the shape!s")
  if fig_size is None:
    width = math.ceil(
      18 * shape[0] / shape[1] * im_list[0].shape[0] / im_list[0].shape[1]
    )
    fig_size = (12, width)
  fig, axs = plt.subplots(shape[0], shape[1], figsize=fig_size)
  axs = axs.flatten()
  fraction = 0.05
  pad = 0.001
  for i in range(len(im_list)):
    im = axs[i].imshow(im_list[i], cmap=cmap)
    if title_array is not None and\
      title_array[i] is not None:
      axs[i].set_title(title_array[i])
    axs[i].axis("off")
    fig.colorbar(
      im, ax=axs[i], fraction=fraction, pad=pad, orientation="horizontal"
    )
  fig.tight_layout(pad=0.0)

  if file_path is not None:
    plt.savefig(file_path, dpi=dpi)
  plt.close()


def jax_memory_profiler(verbose: bool = False):

  all_objects = gc.get_objects()
  total_size = 0
  aux_size = 0
  for obj in all_objects:
    if isinstance(obj, jnp.ndarray):
      total_size += jax.device_get(obj).nbytes
      if jax.device_get(obj).nbytes > 1e7 and\
        'nvidia' in obj.device.device_kind.lower():
        aux_size += jax.device_get(obj).nbytes
        if verbose:
          print(obj.dtype)
          print(obj.shape)
          print(jax.device_get(obj).nbytes)
  print(f"total_size of large array on gpu: {aux_size/1e9:.3f}GB")
  print(f"total_size: {total_size/1e9:.3f}GB")
