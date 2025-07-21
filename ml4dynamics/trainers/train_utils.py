import jax
import jax.numpy as jnp
import numpy as np
import torch

# from dlpack import asdlpack


def run_simulation_fine_grid_correction(
  forward_fn, coarse_model, _iter: callable, label: jnp.ndarray, beta: float,
  res_fn: callable, int_fn: callable, type_: str, x: jnp.ndarray
):
  r"""
  TODO: Come up with a better name, this is more or less a synthetic
  scenario.

  The simulation is performed over the fine grid $2n$, but in each step instead
  of advancing using the expensive fine-grid solver, we:
    1. restrict the fine-grid variable onto coarse grid;
    2. advance the coarse-grid variable using cheap coarse-grid solver;
    3. interpolate the updated coarse-grid variable to onto the fine grid;
    4. add a correction to it

  $$
    u_{k+1}^{2n} = I_n^{2n} \circ f_n(R_{2n}^{n}(u_k^{2n})) + y_k^{2n}
  $$

  Following the DAgger paper, we use beta to mix the current policy with the
  expert policy to stabilize the trajectory
  
  """

  @jax.jit
  def iter(x: jnp.array, expert: jnp.array = 0):
    x_res = res_fn(x)
    x_res = _iter(x_res)
    x_next = int_fn(x_res)
    if type_ == "pad":
      if x.ndim == 1:
        x = jnp.concatenate([x, jnp.zeros(1)], axis=0)
      elif x.ndim == 2:
        x = jnp.concatenate([x, jnp.zeros((1, x.shape[1]))], axis=0)
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    correction = forward_fn(x.reshape(1, *x.shape))
    
    if type_ == "pad":
      if correction.ndim == 2:
        correction = correction[:, :-1]
      elif correction.ndim == 3:
        correction = correction[:, :-1, :]
    
    return x_next + (correction[0] * (1 - beta) + beta * expert) * dt

  dt = coarse_model.dt
  step_num = coarse_model.step_num
  x_hist = np.zeros([step_num, *x.shape])
  for i in range(step_num):
    x_hist[i] = x
    x = iter(x, label[i])

  return x_hist


def run_simulation_fine_grid_correction_torch(
  model, rd_fine, coarse_model, r: int, device, x: jnp.ndarray
):

  nx = x.shape[1]
  dt = coarse_model.dt
  step_num = rd_fine.step_num
  x_hist = jnp.zeros([step_num, 2, nx, nx])
  for i in range(step_num):
    x_hist = x_hist.at[i].set(x)
    tmp = (
      x[:, 0::2, 0::2] + x[:, 1::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 1::2]
    ) / 4
    x = coarse_model.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
    x = np.vstack(
      [
        np.kron(x[0], np.ones((r, r))).reshape(1, nx, nx),
        np.kron(x[1], np.ones((r, r))).reshape(1, nx, nx),
      ]
    )

    # naive jax-torch data exchange from numpy, gpu-cpu
    # x_np = np.asarray(x)
    # x_torch = torch.from_numpy(x_np).clone().to(device)
    # correction = model(x_torch.reshape((1, *x.shape)))
    # x += jnp.array((correction[0].detach().cpu().numpy())) * dt

    # jax-torch data exchange via dlpack
    # reference: https://github.com/jax-ml/jax/issues/1100
    correction = model(
      torch.from_dlpack(asdlpack(x)).reshape((1, *x.shape)).to(device)
    )
    x += jnp.array(jnp.from_dlpack(asdlpack(correction[0].detach()))) * dt

  return x_hist


def run_simulation_coarse_grid_correction(
  forward_fn, model, _iter: callable, label: jnp.ndarray, beta: float,
  type_: str, x: jnp.ndarray
):

  @jax.jit
  def iter(x: jnp.array, expert: jnp.array = 0):
    x_next = _iter(x)
    if type_ == "pad":
      if x.ndim == 1:
        x = jnp.concatenate([x, jnp.zeros(1)], axis=0)
      elif x.ndim == 2:
        x = jnp.concatenate([x, jnp.zeros((1, x.shape[1]))], axis=0)
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    correction = forward_fn(x.reshape(1, *x.shape))
    if type_ == "pad":
      if correction.ndim == 2:
        correction = correction[:, :-1]
      elif correction.ndim == 3:
        correction = correction[:, :-1, :]
    
    return x_next + (correction[0] * (1 - beta) + beta * expert) * dt

  dt = model.dt
  step_num = model.step_num
  x_hist = np.zeros([step_num, *x.shape])
  for i in range(step_num):
    x_hist[i] = x
    x = iter(x, label[i])

  return x_hist


def run_simulation_sgs(
  forward_fn, model, _iter: callable, label: jnp.ndarray, beta: float,
  type_: str, x: jnp.ndarray
):

  @jax.jit
  def iter(x: jnp.array, expert: jnp.array = 0):
    x_next = _iter(x)
    if type_ == "pad":
      if x.ndim == 1:
        x = jnp.concatenate([x, jnp.zeros(1)], axis=0)
      elif x.ndim == 2:
        x = jnp.concatenate([x, jnp.zeros((1, x.shape[1]))], axis=0)
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    correction = forward_fn(x.reshape(1, *x.shape))
    if type_ == "pad":
      if correction.ndim == 2:
        correction = correction[:, :-1]
      elif correction.ndim == 3:
        correction = correction[:, :-1, :]
    
    tmp = correction[0] * dx**2
    if model.model_type == "KS":
      # tmp = (jnp.roll(correction[0], -1) - jnp.roll(correction[0], 1)) / 2 / dx
      # tmp = tmp.at[0].set(correction[0, 1] / 2 / dx)
      # tmp = tmp.at[-1].set(-correction[0, -2] / 2 / dx)
      tmp = model.L1 @ correction[0]
    return x_next + (tmp * (1 - beta) + beta * expert) * dt

  dt = model.dt
  dx = model.dx
  step_num = model.step_num
  x_hist = np.zeros([step_num, *x.shape])
  for i in range(step_num):
    x_hist[i] = x
    x = iter(x, label[i])

  return x_hist


def run_ns_simulation_pressue_correction(
  forward_fn, ns_model, label: jnp.ndarray, beta: float, x: jnp.ndarray
):

  x = jnp.array(x)
  step_num = ns_model.step_num
  x_hist = np.zeros([step_num, *x.shape])
  for i in range(step_num):
    x_hist[i] = x
    correction = forward_fn(x.reshape(1, *x.shape))
    u, v, _ = ns_model.projection_correction(
      x[..., 0], x[..., 1], correction[0, ..., 0], correction=True
    )
    x = jnp.concatenate([u[..., None], v[..., None]], axis=-1)

  return x_hist
