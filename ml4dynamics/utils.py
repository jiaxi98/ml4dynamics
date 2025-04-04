import copy
import gc
from functools import partial

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import ml_collections
import numpy as np
import numpy.linalg as nalg
import optax
import torch
from time import time
from box import Box
from jax import random as random
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ml4dynamics import dynamics
from ml4dynamics.models.models_jax import CustomTrainState, UNet
from ml4dynamics.trainers import train_utils
from ml4dynamics.types import PRNGKey

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)


def load_data(
  config_dict: ml_collections.ConfigDict,
  batch_size: int,
  num_workers: int = 16,
  mode: str = "jax"
):

  config = Box(config_dict)
  pde_type = config.name
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  case_num = config.sim.case_num
  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"

  with h5py.File(h5_filename, "r") as h5f:
    inputs = h5f["data"]["inputs"][()]
    outputs = h5f["data"]["outputs"][()]
    if mode == "jax":
      # shape = (*, nx, nx, 2)
      inputs = inputs.transpose(0, 2, 3, 1)
      outputs = outputs.transpose(0, 2, 3, 1)
    if mode == "torch":
      GPU = 0
      device = torch.device(
        "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
      )
      inputs = torch.from_numpy(inputs).to(device)
      outputs = torch.from_numpy(outputs).to(device)
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
  return inputs, outputs, train_dataloader, test_dataloader


def create_fine_coarse_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  d = config.react_diff.d
  T = config.react_diff.T
  dt = config.react_diff.dt
  Lx = config.react_diff.Lx
  nx = config.react_diff.nx
  r = config.react_diff.r
  rd_fine = dynamics.RD(
    L=Lx,
    N=nx**2 * 2,
    T=T,
    dt=dt,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
  )
  rd_coarse = dynamics.RD(
    L=Lx,
    N=(nx // r)**2 * 2,
    T=T,
    dt=dt,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
  )
  return rd_fine, rd_coarse


def prepare_unet_train_state(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  nx = config.react_diff.nx
  rng = random.PRNGKey(config.sim.seed)
  unet = UNet(2)
  rng1, rng2 = random.split(rng)
  init_rngs = {'params': rng1, 'dropout': rng2}
  unet_variables = unet.init(init_rngs, jnp.ones([1, nx, nx, 2]))
  step_per_epoch = 4000 * config.sim.case_num // config.train.batch_size_unet
  # TODO: need to specify the scheduler here for different training
  schedule = optax.piecewise_constant_schedule(
    init_value=config.train.lr,
    boundaries_and_scales={
      int(b): 0.1
      for b in jnp.arange(
        20 * step_per_epoch,
        config.train.epochs * step_per_epoch,
        20 * step_per_epoch)
    }
  )
  optimizer = optax.adam(schedule)
  train_state = CustomTrainState.create(
    apply_fn=unet.apply,
    params=unet_variables["params"],
    tx=optimizer,
    batch_stats=unet_variables["batch_stats"]
  )
  return train_state, schedule


def eval_a_priori(
  train_state: CustomTrainState,
  train_dataloader: DataLoader,
  test_dataloader: DataLoader,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
):
  
  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in train_dataloader:
    predict, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(batch_inputs),
      is_training=False
    )
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"train loss: {total_loss/count:.4e}")
  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in test_dataloader:
    predict, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(batch_inputs),
      is_training=False
    )
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"test loss: {total_loss/count:.4e}")

  # visualization
  n_plot = 6
  fig, axs = plt.subplots(3, n_plot, figsize=(12, 6))
  index_array = [0, 800, 1600, 2400, 3200, 4000]
  for j in range(n_plot):
    predict, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(inputs[index_array[j]:index_array[j]+1]),
      is_training=False
    )
    # use cm.twilight to emphasize the middle range
    # use cm.coolwarm to emphasize the extreme range
    im = axs[0, j].imshow(outputs[index_array[j], ..., 0], cmap=cm.twilight)
    _ = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
    axs[0, j].axis("off")
    im = axs[1, j].imshow(predict[0, ..., 0], cmap=cm.twilight)
    _ = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
    axs[1, j].axis("off")
    im = axs[2, j].imshow(
      outputs[index_array[j], ..., 0] - predict[0, ..., 0],
      cmap=cm.twilight
    )
    _ = fig.colorbar(im, ax=axs[2, j], orientation='horizontal')
    axs[2, j].axis("off")

  plt.savefig(f"results/fig/apriori.pdf")
  plt.clf()

  # visualize the dataset
  n_plot = 6
  fig, axs = plt.subplots(2, n_plot, figsize=(12, 5))
  index_array = [0, 800, 1600, 2400, 3200, 4000]
  for j in range(n_plot):
    im = axs[0, j].imshow(inputs[index_array[j], ..., 0])
    _ = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
    axs[0, j].axis("off")
    im = axs[1, j].imshow(outputs[index_array[j], ..., 0])
    _ = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
    axs[1, j].axis("off")
  plt.savefig("results/fig/dataset.pdf")


def eval_a_posteriori(
  config_dict: ml_collections.ConfigDict,
  train_state: CustomTrainState,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
  fig_name: str = "cloudmap",
):
  
  config = Box(config_dict)
  nx = config.react_diff.nx
  r = config.react_diff.r
  dt = config.react_diff.dt
  _, rd_coarse = create_fine_coarse_simulator(config)
  beta = 0
  run_simulation = partial(
    train_utils.run_simulation_coarse_grid_correction, train_state, rd_coarse,
    outputs, nx, r, dt, beta
  )
  start = time()
  x_hist = run_simulation(inputs[0].transpose(2, 0, 1))
  step_num = rd_coarse.step_num
  print(f"simulation takes {time() - start:.2f}s...")
  if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
    print("similation contains NaN!")
    breakpoint()
  print(
    "L2 error: {:.4e}".format(
      np.sum(
        np.linalg.norm(
          x_hist.reshape(step_num, -1) -
          (inputs.transpose(0, 3, 1, 2)).reshape(step_num, -1),
          axis=1
        )
      )
    )
  )

  # visualization
  n_plot = 6
  fig, axs = plt.subplots(3, n_plot, figsize=(12, 6))
  index_array = [0, 800, 1600, 2400, 3200, 4000]
  for j in range(n_plot):
    im = axs[0, j].imshow(inputs[index_array[j], ..., 0], cmap=cm.twilight)
    _ = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
    axs[0, j].axis("off")
    im = axs[1, j].imshow(x_hist[index_array[j], 0], cmap=cm.twilight)
    _ = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
    axs[1, j].axis("off")
    im = axs[2, j].imshow(
      inputs[index_array[j], ..., 0] - x_hist[index_array[j], 0],
      cmap=cm.twilight
    )
    _ = fig.colorbar(im, ax=axs[2, j], orientation='horizontal')
    axs[2, j].axis("off")

  plt.savefig(f"results/fig/{fig_name}.pdf")
  plt.clf()


###############################################################################
#                   Numerical solver of the reaction-diffusion equation:
# For linear term, we use different discretization scheme, e.g. explicit,
#  implicit, Crank-Nielson, ADI etc.
# For nonlinear term, we use the explicit scheme (need to implement Ashford)
###############################################################################


def assembly_RDmatrix(n, dt, dx, beta=1.0, gamma=0.05, d=2):
  """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
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

    u = jax.scipy.linalg.solve(L_uminus, rhsu)
    u = jax.scipy.linalg.solve(L_uminus, u.T)
    u = u.T
    v = jax.scipy.linalg.solve(L_vminus, rhsv)
    v = jax.scipy.linalg.solve(L_vminus, v.T)
    v = v.T
    return u, v

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
    while nalg.norm(res) > tol:
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


def assembly_NSmatrix(nx, ny, dt, dx, dy):
  """assemble matrices used in the calculation
    LD: Laplacian operator with Dirichlet BC
    LN: Laplacian operator with Neuman BC, notice that this operator may have
    different form depends on the position of the boundary, here we use the
    case that boundary is between the outmost two grids
    L:  Laplacian operator associated with current BC with three Neuman BCs on
    upper, lower, left boundary and a Dirichlet BC on right
    """

  global L
  LNx = np.eye(nx) * (-2)
  LNy = np.eye(ny) * (-2)
  for i in range(1, nx - 1):
    LNx[i, i - 1] = 1
    LNx[i, i + 1] = 1
  for i in range(1, ny - 1):
    LNy[i, i - 1] = 1
    LNy[i, i + 1] = 1
  LNx[0, 1] = 1
  LNx[0, 0] = -1
  LNx[-1, -1] = -1
  LNx[-1, -2] = 1
  LNy[0, 1] = 1
  LNy[0, 0] = -1
  LNy[-1, -1] = -1
  LNy[-1, -2] = 1
  # LNx = spa.csc_matrix(LNx / (dx**2))
  # LNy = spa.csc_matrix(LNy / (dy**2))
  LNx = LNx / (dx**2)
  LNy = LNy / (dy**2)
  # BE CAREFUL, SINCE THE LAPLACIAN MATRIX IN X Y DIRECTION IS NOT THE SAME
  #L2N = spa.kron(LNy, spa.eye(nx)) + spa.kron(spa.eye(ny), LNx)
  # L2N = spa.kron(LNx, spa.eye(ny)) + spa.kron(spa.eye(nx), LNy)
  L2N = np.kron(LNx, np.eye(ny)) + np.kron(np.eye(nx), LNy)
  L = copy.deepcopy(L2N)
  #for i in range(ny):
  #    L[(i+1)*nx - 1, (i+1)*nx - 1] = L[(i+1)*nx - 1, (i+1)*nx - 1] - 2
  for i in range(ny):
    L[-1 - i, -1 - i] = L[-1 - i, -1 - i] - 2 / (dx**2)

  return


def projection_correction(
  u: jnp.ndarray,
  v: jnp.ndarray,
  t,
  dx=1 / 32,
  dy=1 / 32,
  nx=128,
  ny=32,
  y0=0.325,
  eps=1e-7,
  dt=.01,
  Re=100,
  flag=True
):
  """projection method to solve the incompressible NS equation
    The convection discretization is given by central difference
    u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx"""

  # central difference for first derivative
  u_x = (u[2:, 1:-1] - u[:-2, 1:-1]) / dx / 2
  u_y = (u[1:-1, 2:] - u[1:-1, :-2]) / dy / 2
  v_x = (v[2:, 1:-1] - v[:-2, 1:-1]) / dx / 2
  v_y = (v[1:-1, 2:] - v[1:-1, :-2]) / dy / 2

  # five pts scheme for Laplacian
  u_xx = (-2 * u[1:-1, 1:-1] + u[2:, 1:-1] + u[:-2, 1:-1]) / (dx**2)
  u_yy = (-2 * u[1:-1, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) / (dy**2)
  #u_xy = (u[2:,2:]+u[:-2,:-2]-2*u[1:-1,1:-1])/(dx**2)/2 - \
  #        (u_xx+u_yy)/2
  v_xx = (-2 * v[1:-1, 1:-1] + v[2:, 1:-1] + v[:-2, 1:-1]) / (dx**2)
  v_yy = (-2 * v[1:-1, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2]) / (dy**2)
  #v_xy = (v[2:,2:]+v[:-2,:-2]-2*v[1:-1,1:-1])/(dx**2)/2 - \
  #        (v_xx+v_yy)/2

  # interpolate u, v on v, u respectively, we interpolate using the four neighbor nodes
  u2v = (u[:-2, 1:-2] + u[1:-1, 1:-2] + u[:-2, 2:-1] + u[1:-1, 2:-1]) / 4
  v2u = (v[1:-1, :-1] + v[2:, :-1] + v[1:-1, 1:] + v[2:, 1:]) / 4

  # prediction step: forward Euler
  u[1:-1, 1:-1] = u[
    1:-1, 1:-1] + dt * ((u_xx + u_yy) / Re - u[1:-1, 1:-1] * u_x - v2u * u_y)
  v[1:-1, 1:-1] = v[
    1:-1, 1:-1] + dt * ((v_xx + v_yy) / Re - u2v * v_x - v[1:-1, 1:-1] * v_y)

  # correction step: calculating the residue of Poisson equation as the
  #  divergence of new velocity field
  divu = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
  # GMRES with initial guess pressure in last time step
  # p = jsla.gmres(A=L, b=divu.reshape(nx * ny), x0=p).reshape([nx, ny])
  # BiSTABCG with initial guess pressure in last time step
  # p = jsla.bicgstab(A=L, b=divu.reshape(nx * ny), x0=p).reshape([nx, ny])
  p = jax.scipy.linalg.solve(L, divu.reshape(nx * ny)).reshape([nx, ny])

  u[1:-2, 1:-1] = u[1:-2, 1:-1] - (p[1:, :] - p[:-1, :]) / dx
  v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:, 1:] - p[:, :-1]) / dy
  u[-2, 1:-1] = u[-2, 1:-1] + 2 * p[-1, :] / dx

  # check the corrected velocity field is divergence free
  divu = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
  if flag and nalg.norm(divu) > eps:
    print(nalg.norm(divu))
    print(t)
    print("Velocity field is not divergence free!!!")
    flag = False

  # update Dirichlet BC on left, upper, lower boundary
  u[:, 0] = -u[:, 1]
  u[:, -1] = -u[:, -2]
  v[0, 1:-1] = 2 * np.exp(-50 * (np.linspace(dy, 1 - dy, ny - 1) - y0)**2
                          ) * np.sin(t) - v[1, 1:-1]
  # update Neuman BC on right boundary
  u[-1, :] = u[-3, :]
  v[-1, :] = v[
    -2, :]  # alternative choice to use Neuman BC for v on the right boundary
  #v[-1, 1:-1] = v[-1, 1:-1] + (p[-1, 1:] - p[-1, :-1])/dy

  return u, v, p / dt, flag


def a_posteriori_analysis(
  config: ml_collections.ConfigDict,
  ks_fine,
  ks_coarse,
  correction_nn: callable,
  params: hk.Params,
):

  c = config.ks.c
  L = config.ks.L
  T = config.ks.T
  init_scale = config.ks.init_scale
  BC = config.ks.BC
  # solver parameters
  if BC == "periodic":
    N1 = config.ks.nx
  elif BC == "Dirichlet-Neumann":
    N1 = config.ks.nx - 1
  r = config.ks.r
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
      # local model: [1] to [1]
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
      # local model: [1] to [1]
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
      # local model: [1] to [1]
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
  from ml4dynamics.visualize import plot_stats
  plot_stats(
    np.arange(ks_fine.x_hist.shape[0]) * ks_fine.dt,
    ks_fine.x_hist,
    baseline,
    correction1,
    correction2,
    f"results/fig/ks_c{c}T{T}n{config.sim.case_num}_{train_mode}_stats.pdf",
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
  im_array, fig_size=(10, 4), title_array=None, file_path=None
):

  fig, axs = plt.subplots(
    im_array.shape[0], im_array.shape[1], figsize=fig_size
  )
  axs = axs.flatten()
  im = []
  fraction = 0.05
  pad = 0.001
  cbar = []
  for i in range(im_array.shape[0]):
    for j in range(im_array.shape[1]):
      im.append(
        axs[i * im_array.shape[1] + j].imshow(im_array[i, j], cmap=cm.viridis)
      )
      if title_array is not None:
        axs[i * im_array.shape[1] +
            j].set_title(title_array[i * im_array.shape[1] + j])
      axs[i * im_array.shape[1] + j].axis("off")
      cbar.append(
        fig.colorbar(
          im[-1],
          ax=axs[i * im_array.shape[1] + j],
          fraction=fraction,
          pad=pad,
          orientation="horizontal"
        )
      )
  fig.tight_layout(pad=0.0)

  if file_path is not None:
    plt.savefig(file_path, dpi=300)
  plt.clf()


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
