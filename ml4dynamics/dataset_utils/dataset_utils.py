import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from box import Box


def calc_correction(rd_fine, rd_coarse, nx: float, r: int, uv: jnp.ndarray):
  """
  Args:
    uv: shape = [nx, nx, 2]
  """
  next_step_fine = rd_fine.adi(uv)
  tmp = jnp.zeros((nx // r, nx // r, 2))
  for k in range(r):
    for j in range(r):
      tmp += uv[k::r, j::r]
  tmp = tmp / (r**2)
  next_step_coarse = rd_coarse.adi(tmp)
  next_step_coarse_interp = jnp.concatenate(
    [
      jnp.kron(next_step_coarse[..., 0], jnp.ones((r, r)))[..., None],
      jnp.kron(next_step_coarse[..., 1], jnp.ones((r, r)))[..., None],
    ],
    axis=2
  )

  return next_step_fine - next_step_coarse_interp


def create_box_filter(N1: int, N2: int, r: int, BC: str, s: int):
  """Create box filter (averaging) operator
  
  Args:
    N1: fine grid size
    N2: coarse grid size  
    r: coarsening ratio
    BC: boundary condition ("periodic" or "Dirichlet-Neumann")
    s: stencil size (must be odd)
  """
  if s % 2 == 0:
    raise ValueError("Stencil size must be odd")
    
  res_op = jnp.zeros((N2, N1))
  half_stencil = s // 2
  
  if r == 2:
    # Box filter for r=2 with variable stencil size
    if BC == "periodic":
      for i in range(N2):
        start = i * r
        for j in range(s):
          idx = (start + j) % N1  # wrap around domain
          res_op = res_op.at[i, idx].set(1)
      res_op /= s / r  
    else:  # Dirichlet-Neumann - ignore for now
      raise Exception("Dirichlet-Neumann not implemented for r=2")
      
  elif r == 4:
    if BC == "periodic":
      for i in range(N2):
        start = i * r
        for j in range(s):
          idx = (start + j) % N1  # wrap around domain
          res_op = res_op.at[i, idx].set(1)
      res_op /= s / 4  # normalize to preserve energy scale
    else:  # Dirichlet-Neumann
      for i in range(N2):
        res_op = res_op.at[i, i * r:i * r + s].set(1)
      res_op /= s / 4
      
  elif r == 8:
    # Box filter for r=8 with variable stencil size
    if BC == "periodic":
      for i in range(N2):
        start = i * r
        for j in range(s):
          idx = (start + j) % N1  # wrap around domain
          res_op = res_op.at[i, idx].set(1)
      res_op /= s / r  
    else:  # Dirichlet-Neumann - ignore for now
      raise Exception("Dirichlet-Neumann not implemented for r=8")

  elif r == 16:
    if BC == "periodic":
      for i in range(N2):
        start = i * r
        for j in range(s):
          idx = (start + j) % N1  # wrap around domain
          res_op = res_op.at[i, idx].set(1)
      res_op /= s / r  # normalize to preserve energy scale
    else:  # Dirichlet-Neumann
      raise Exception("Dirichlet-Neumann not implemented for r=16")
  
  return res_op


def res_int_fn(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  r = config.sim.rx
  if config.case == "ks":
    BC = config.sim.BC
    s = config.sim.stencil_size  # Get stencil size from config
    
    if BC == "periodic":
      N1 = config.sim.n
    elif BC == "Dirichlet-Neumann":
      N1 = config.sim.n - 1
    N2 = N1 // r
    
    # Create box filter using adaptive stencil size
    res_op = create_box_filter(N1, N2, r, BC, s)
    res_op /= r
    int_op = jnp.linalg.pinv(res_op)
    assert jnp.allclose(res_op @ int_op, jnp.eye(N2))
    assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2))

    @jax.jit
    def res_fn(x):
      return (x.reshape(-1, N1) @ res_op.T).reshape(N2, -1)

    @jax.jit
    def int_fn(x):
      return (x.reshape(-1, N2) @ int_op.T).reshape(N1, -1)
  elif config.case == "react_diff" or config.case == "ns_hit":
    n = config.sim.n

    @jax.jit
    def res_fn(x):
      result = jnp.zeros((n // r, n // r, x.shape[-1]))
      for k in range(r):
        for j in range(r):
          result += x[k::r, j::r]
      return result / (r**2)

    @jax.jit
    def int_fn(x):
      # only works for the case with 2 components
      return jnp.concatenate(
        [
          jnp.kron(x[..., 0], jnp.ones((r, r)))[..., None],
          jnp.kron(x[..., 1], jnp.ones((r, r)))[..., None],
        ],
        axis=2
      )

  return res_fn, int_fn
